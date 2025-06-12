from flask import Blueprint, request, jsonify
from .utils import get_collection, get_chat_collection  
import numpy as np
from datetime import datetime

def average_embedding(embedding_dicts):
    vectors = [e["embedding"] for e in embedding_dicts if "embedding" in e]
    if not vectors:
        return []
    return np.mean(np.array(vectors), axis=0).tolist()

category_map = {
    "top": ["kurta", "shirt", "top", "blouse", "tshirt", "salwar", "kameez", "tops", "upper", "topwear"],
    "bottom": ["jeans", "palazzo", "skirt", "churidar", "pants", "trousers", "leggings", "bottomwear"],
    "shoes": ["sandals", "heels", "shoes", "flats", "juttis", "jutti", "footwear", "sneakers", "slidders"],
    "accessory": ["earrings", "necklace", "watch", "bangles", "belt", "bag", "jewelry", "accessory", "accessories"]
}

def get_canonical_category(raw_category):
    if not isinstance(raw_category, str):
        return ""
    return (raw_category or "").replace("-", "").replace("_", "").replace(" ", "").lower()

# Map cleaned category to standard category bucket
def map_to_canonical_category(raw_category):
    if not isinstance(raw_category, str):
        return "unknown"

    clean = get_canonical_category(raw_category)

    for canonical, keywords in category_map.items():
        if any(keyword in clean for keyword in keywords):
            return canonical

    return "unknown"

def find_similar_by_category(collection, target, target_cat, top_k=5):
    pipeline = [
        {
            "$match": {
                "canonical_category": target_cat
            }
        },
        { "$limit": top_k }
    ]
    return list(collection.aggregate(pipeline))


service_bp = Blueprint('service', __name__)

# Route 1: Add single entry
@service_bp.route('/add_entry', methods=['POST'])
def add_entry():
    data = request.get_json()
    required_fields = ["company_id", "slug", "category", "gender", "embeddings"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    company_id = data["company_id"]
    collection = get_collection(company_id)

    entry = {
        "slug": data["slug"],
        "category": data["category"],
        "canonical_category": map_to_canonical_category(data["category"]),
        "gender": data["gender"],
        "embeddings": data["embeddings"],
        "average_embedding": average_embedding(data["embeddings"])
    }

    collection.insert_one(entry)
    return jsonify({"message": "Entry added successfully"}), 201

# Route 2: Get all entries
@service_bp.route('/get_entries/<company_id>', methods=['GET'])
def get_entries(company_id):
    collection = get_collection(company_id)
    entries = list(collection.find({}, {"_id": 0}))
    return jsonify({"entries": entries}), 200

# Route 3: Add multiple entries
@service_bp.route('/add_multiple_entries', methods=['POST'])
def add_multiple_entries():
    data = request.get_json()

    if "company_id" not in data or "products" not in data:
        return jsonify({"error": "Missing company_id or products"}), 400

    company_id = data["company_id"]
    products = data["products"]

    collection = get_collection(company_id)

    # Validate each product
    for product in products:
        required_fields = ["slug", "category", "gender", "embeddings"]
        if not all(field in product for field in required_fields):
            return jsonify({"error": "Each product must have required fields"}), 400

    # Prepare entries for insertion
    entries = []
    for product in products:
        entry = {
            "slug": product["slug"],
            "category": product["category"],
            "gender": product["gender"],
            "embeddings": product["embeddings"]
        }
        entries.append(entry)

    if entries:
        collection.insert_many(entries)
        return jsonify({"message": f"{len(entries)} entries added successfully"}), 201
    else:
        return jsonify({"message": "No valid entries to insert"}), 200
    

@service_bp.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    slug = data.get("slug")
    company_id = data.get("company_id")
    top_k = data.get("top_k", 3)

    if not slug or not company_id:
        return jsonify({"error": "slug and company_id are required"}), 400

    collection = get_collection(company_id)

    # 1️⃣ Fetch target product
    target = collection.find_one(
        {"slug": slug},
        {"_id": 0, "average_embedding": 1, "canonical_category": 1}
    )

    if not target or "average_embedding" not in target:
        return jsonify({"error": "Target product not found or missing average_embedding"}), 404

    target_embedding = target["average_embedding"]
    target_category = target.get("canonical_category", "unknown")
    target_norm = np.linalg.norm(target_embedding)

    VECTOR_DIM = len(target_embedding)

    # 2️⃣ Recommendation pipeline: only same canonical_category
    pipeline = [
        {
            "$match": {
                "slug": { "$ne": slug },
                "canonical_category": target_category
            }
        },
        { "$addFields": {
            "dotProduct": {
                "$reduce": {
                    "input": { "$range": [0, VECTOR_DIM] },
                    "initialValue": 0,
                    "in": {
                        "$add": [
                            "$$value",
                            {
                                "$multiply": [
                                    { "$arrayElemAt": ["$average_embedding", "$$this"] },
                                    { "$arrayElemAt": [ { "$literal": target_embedding }, "$$this" ] }
                                ]
                            }
                        ]
                    }
                }
            },
            "norm": {
                "$sqrt": {
                    "$reduce": {
                        "input": { "$range": [0, VECTOR_DIM] },
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {
                                    "$multiply": [
                                        { "$arrayElemAt": ["$average_embedding", "$$this"] },
                                        { "$arrayElemAt": ["$average_embedding", "$$this"] }
                                    ]
                                }
                            ]
                        }
                    }
                }
            },
            "targetNorm": target_norm
        }},
        { "$addFields": {
            "cosineSimilarity": {
                "$divide": [
                    "$dotProduct",
                    { "$multiply": [ "$norm", "$targetNorm" ] }
                ]
            }
        }},
        { "$sort": { "cosineSimilarity": -1 } },
        { "$limit": top_k },
        { "$project": { "_id": 0, "slug": 1, "category": 1, "gender": 1, "cosineSimilarity": 1 } }
    ]

    results = list(collection.aggregate(pipeline))

    return jsonify({ "recommendations": results }), 200

@service_bp.route('/add_average_embedding', methods=['POST'])
def add_average_embedding():
    data = request.get_json()
    company_id = data.get("company_id")

    if not company_id:
        return jsonify({"error": "company_id is required"}), 400

    collection = get_collection(company_id)

    # Find products missing "average_embedding"
    products_missing_avg = collection.find(
        { "average_embedding": { "$exists": False } },
        { "_id": 1, "embeddings": 1, "slug": 1 }
    )

    updated_count = 0
    for product in products_missing_avg:
        embeddings = product.get("embeddings", [])
        avg_emb = average_embedding(embeddings)

        # Update the document
        collection.update_one(
            { "_id": product["_id"] },
            { "$set": { "average_embedding": avg_emb } }
        )
        updated_count += 1

    return jsonify({
        "message": f"Added average_embedding to {updated_count} products in company {company_id}."
    }), 200

@service_bp.route('/bundle_outfit', methods=['POST'])
def bundle_outfit():
    data = request.get_json()
    slug = data.get("slug")
    company_id = data.get("company_id")

    if not slug or not company_id:
        return jsonify({"error": "Missing slug or company_id"}), 400

    collection = get_collection(company_id)

    # Step 1: Get the target product (only needed fields)
    target = collection.find_one({"slug": slug}, {"slug": 1, "category": 1, "gender": 1, "embeddings": 1})
    if not target:
        return jsonify({"error": "Target product not found"}), 404

    # Step 2: Compute canonical category
    target_category = map_to_canonical_category(target.get("category") or target.get("name"))

    # Step 3: Define complementary categories
    complementary_categories = {
        "top": ["bottom", "shoes", "accessory"],
        "bottom": ["top", "shoes", "accessory"],
        "shoes": ["top", "bottom", "accessory"],
        "accessory": ["top", "bottom", "shoes"],
        "unknown": ["top", "bottom", "shoes", "accessory"]
    }.get(target_category, [])

    # Step 4: Find similar products for each complementary category
    bundled = []
    for cat in complementary_categories:
        similar_raw = find_similar_by_category(collection, target, cat, top_k=5)

        # Only keep slug, category, and gender
        similar = [
            {
                "slug": prod.get("slug"),
                "category": prod.get("category"),
                "gender": prod.get("gender")
            } for prod in similar_raw
        ]

        bundled.append({
            "category": cat,
            "products": similar
        })

    # Final minimal target
    minimal_target = {
        "slug": target.get("slug"),
        "category": target.get("category"),
        "gender": target.get("gender")
    }

    return jsonify({
        "target": minimal_target,
        "bundled_outfit": bundled
    }), 200

@service_bp.route('/add_canonical_category', methods=['POST'])
def add_canonical_category():
    data = request.get_json()
    company_id = data.get("company_id")

    if not company_id:
        return jsonify({"error": "company_id is required"}), 400

    collection = get_collection(company_id)

    # Find products missing "canonical_category"
    products_missing_canonical = collection.find(
        { "canonical_category": { "$exists": False } },
        { "_id": 1, "category": 1, "name": 1, "slug": 1 }
    )

    updated_count = 0
    for product in products_missing_canonical:
        raw_category = product.get("category") or product.get("name")
        canonical_cat = map_to_canonical_category(raw_category)

        collection.update_one(
            { "_id": product["_id"] },
            { "$set": { "canonical_category": canonical_cat } }
        )
        updated_count += 1

    return jsonify({
        "message": f"Added canonical_category to {updated_count} products in company {company_id}."
    }), 200


@service_bp.route('/bulk_recommend', methods=['POST'])
def bulk_recommend():
    data = request.get_json()
    embeddings = data.get("embeddings", [])
    company_id = data.get("company_id")

    if not embeddings or not company_id:
        return jsonify({"error": "Missing embeddings or company_id"}), 400

    collection = get_collection(company_id)

    avg_embedding = average_embedding(embeddings)
    if not avg_embedding:
        return jsonify({"error": "No valid embeddings found"}), 400

    norm = np.linalg.norm(avg_embedding)
    dim = len(avg_embedding)

    categories = ["top", "bottom", "shoes", "accessory"]
    result = {}

    for cat in categories:
        pipeline = [
            {
                "$match": {
                    "canonical_category": cat
                }
            },
            {
                "$addFields": {
                    "dotProduct": {
                        "$reduce": {
                            "input": { "$range": [0, dim] },
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {
                                        "$multiply": [
                                            { "$arrayElemAt": ["$average_embedding", "$$this"] },
                                            { "$arrayElemAt": [ { "$literal": avg_embedding }, "$$this" ] }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    "norm": {
                        "$sqrt": {
                            "$reduce": {
                                "input": { "$range": [0, dim] },
                                "initialValue": 0,
                                "in": {
                                    "$add": [
                                        "$$value",
                                        {
                                            "$multiply": [
                                                { "$arrayElemAt": ["$average_embedding", "$$this"] },
                                                { "$arrayElemAt": ["$average_embedding", "$$this"] }
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "targetNorm": norm
                }
            },
            {
                "$addFields": {
                    "cosineSimilarity": {
                        "$divide": [
                            "$dotProduct",
                            { "$multiply": [ "$norm", "$targetNorm" ] }
                        ]
                    }
                }
            },
            { "$sort": { "cosineSimilarity": -1 } },
            { "$limit": 1 }, #edit this
            { "$project": { "slug": 1, "_id": 0 } }
        ]

        products = list(collection.aggregate(pipeline))
        result[cat] = [p["slug"] for p in products]

    return jsonify(result), 200

# Add this new route to your existing service_bp blueprint
@service_bp.route('/chatdata', methods=['POST'])
def chatpost():
    data = request.get_json()
    query = data.get("query", "").strip()
    company_id = data.get("company_id")

    # Validate required fields
    if not query:
        return jsonify({"error": "Missing 'query' in JSON body"}), 400
    
    if not company_id:
        return jsonify({"error": "Missing 'company_id' in JSON body"}), 400
    
    try:
        # Get the chat collection for the specific company
        chats_collection = get_chat_collection(company_id)
        
        # Prepare the chat document following the schema
        chat_document = {
            "query": query,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }

        # Insert the document
        result = chats_collection.insert_one(chat_document)
        
        return jsonify({
            "message": "Chat data inserted successfully!",
            "inserted_id": str(result.inserted_id),
            "company_id": company_id
        }), 201

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@service_bp.route('/embedding_stats', methods=['POST'])
def embedding_stats():
    data = request.get_json()
    embeddings = data.get("embeddings", [])
    company_id = data.get("company_id")

    if not embeddings or not company_id:
        return jsonify({"error": "Missing embeddings or company_id"}), 400

    # Calculate average embedding from input
    avg_embedding = average_embedding(embeddings)
    if not avg_embedding:
        return jsonify({"error": "Invalid embeddings provided"}), 400

    # Connect to the appropriate collection
    collection = get_collection(company_id)

    # Retrieve lengths of all stored average embeddings
    products_cursor = collection.find(
        { "average_embedding": { "$exists": True, "$ne": [] } },
        { "_id": 0, "average_embedding": 1 }
    )
    avg_lengths = [len(doc["average_embedding"]) for doc in products_cursor if isinstance(doc.get("average_embedding"), list)]

    return jsonify({
        "input_avg_length": len(avg_embedding),
        "db_avg_embedding_lengths": avg_lengths
    }), 200
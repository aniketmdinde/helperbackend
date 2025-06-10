from flask import Blueprint, request, jsonify
from .utils import get_collection

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
        "gender": data["gender"],
        "embeddings": data["embeddings"]
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
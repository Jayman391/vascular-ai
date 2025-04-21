import os
from flask import Flask, request, jsonify

from src.api.agent import *
from src.api.logging import *

# Initialize Flask app
app = Flask(__name__)

# Global variables to store state
global_response = ""
global_query = ""

# Define the Flask route for the query endpoint
@app.route('/query', methods=['POST'])
def query():
    """Handles incoming user queries and updates global state."""
    global global_query, global_response  # Declare global variables to modify them
    
    # Extract user query from the POST request
    data = request.get_json()
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response = get_response(user_query)

        global_query = user_query  # Update global query
        global_response = response  # Update global response

        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500


@app.route('/rating', methods=['POST'])
def rating():
    """Handles incoming ratings for the responses."""
    data = request.get_json()
    rating = data.get('rating', False)

    if not rating:
        return jsonify({"error": "No rating provided"}), 400
    
    # Assuming 'add_response' stores the rating in some form of database or logging system
    try :
        add_response(
            query=global_query,
            response=global_response,
            rating=rating,
        )
        return jsonify({"status": "Rating received successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to log rating: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

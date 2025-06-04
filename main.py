from app import create_app
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = create_app()
CORS(app, resources={r"*": {"origins": "*"}})

if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
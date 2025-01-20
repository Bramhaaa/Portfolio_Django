from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token

# Import your RAG model setup
from .rag import ai_chain, memory, process_input  # Update with your actual script filename

chat_history = []  # Store chat history for context

@csrf_exempt
@require_http_methods(["POST", "OPTIONS"])
def chatbot_view(request):
    if request.method == "OPTIONS":
        response = JsonResponse({"detail": "CORS preflight"})
        response["Access-Control-Allow-Origin"] = "http://localhost:3000"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        response["Access-Control-Allow-Credentials"] = "true"
        return response

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("input", "")

            if not user_input:
                return JsonResponse({"error": "No input provided"}, status=400)

            # Process input and get the response
            answer = process_input(user_input, chat_history)

            response = JsonResponse({"response": answer})
            response["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response["Access-Control-Allow-Credentials"] = "true"
            return response

        except Exception as e:
            response = JsonResponse({"error": str(e)}, status=500)
            response["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response["Access-Control-Allow-Credentials"] = "true"
            return response
    
    response = JsonResponse({"error": "Invalid request method"}, status=405)
    response["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response["Access-Control-Allow-Credentials"] = "true"
    return response
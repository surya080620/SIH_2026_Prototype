from google import genai

API_KEY = "AQ.Ab8RN6ILSnVyekjdGBt0LCb3oD4iYMlofyT_z6dwiSR2IXFKzQ"

client = genai.Client(api_key=API_KEY)

print("Connecting to Gemini API...")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say: Gemini is connected and working!"
    )
    print("✓ Success! Model Output:")
    print(response.text.strip())
except Exception as e:
    print("Connection failed with error:\n", e)
import config
import openai
import utils.collect_data as cd

MODEL = "gpt-3.5-turbo"
SEP_BARS = "-"*10
DATA_PATH = 'data/'

openai.api_key = config.OPEN_API_KEY

def ask_message():
    print("今日あった出来事を書いてください。嬉しい・悲しいなど、できるだけ感情を含めた表現を使用してください。")
    print("\n")
    print(SEP_BARS + " あなた " + SEP_BARS)
    message = input(">> ")
    print("\n")

    return message

def get_response():
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
            {"role": "system", "content": "You are a friendly counselor."},
            {"role": "user", "content": message}
        ]
    )

    return response

if __name__ == "__main__":
    message = ask_message()
    print(SEP_BARS + " ChatGPT " + SEP_BARS)
    response = get_response()
    answer = response['choices'][0]['message']['content']
    print(answer)

    cd.write_data(cd.extract_emotion_and_cause(message))
    
    
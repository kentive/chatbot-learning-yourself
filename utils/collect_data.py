import config
import tuning_history
import fine_tuning
import openai
import subprocess

MODEL = "gpt-3.5-turbo"
SEP_BARS = "-"*10
DATA_PATH = 'data/'
DATA_TYPE = 'JSONL'
TUNING_THRESHOLD = 5

openai.api_key = config.OPEN_API_KEY

def extract_emotion_and_cause(message):
    INSTRUCTION = f'以下の文章から、複数のできごと(prompt)とそれに伴う感情表現(completion)を抽出し、{DATA_TYPE}形式で出力してください。ただし、{DATA_TYPE}に関係のないテキストは出力しないでください。また、{DATA_TYPE}は1行ずつ改行してください。'
    message = INSTRUCTION + '「' + message.replace('\n', '') + '」'

    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
            {"role": "user", "content": message}
        ]
    )

    if DATA_TYPE == 'CSV':
        res = response['choices'][0]['message']['content'].replace('\n\n', '').replace('prompt,completion\n', '')
    elif DATA_TYPE == 'JSONL':
        res = response['choices'][0]['message']['content'].replace('\n\n', '').replace('[', '').replace(']', '')

    return res

def countLines(file_path):
    line_count = int(subprocess.check_output(['wc', '-l', file_path]).decode().split(' ')[0])

    return line_count

def try_tuning():
    if tuning_history.isProcessing:
        fine_tuning.fine_tune()
        return

    if DATA_TYPE == 'CSV':
        ext = '.csv'
    elif DATA_TYPE == 'JSONL':
        ext = '.jsonl'
    line_count = countLines(DATA_PATH + 'data' + ext)

    if line_count >= TUNING_THRESHOLD * (tuning_history.count + 1):
        fine_tuning.fine_tune()
        tuning_history.count = tuning_history.count + 1

def write_data(message):
    if DATA_TYPE == 'CSV':
        ext = '.csv'
    elif DATA_TYPE == 'JSONL':
        ext = '.jsonl'
    with open(DATA_PATH + 'data' + ext, 'a') as f:
        print(f.write(message + '\n'))

    try_tuning()

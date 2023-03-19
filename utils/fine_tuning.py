import config
import tuning_history
import openai

BASE_MODEL = "ada"
TRAININGF_FILE = 'data/data.jsonl'

openai.api_key = config.OPEN_API_KEY

def fine_tune():
    uploaded_file = openai.File.create(
        file=open(TRAININGF_FILE, "rb"),
        purpose='fine-tune'
    )
    if tuning_history.fine_tuning_job_id:
        fine_tuned_model = openai.FineTune.retrieve(id=tuning_history.fine_tuning_job_id)['fine_tuned_model']
        if fine_tuned_model == None:
            tuning_history.isProcessing = True
            return
        else:
            ft = openai.FineTune.create(training_file=uploaded_file['id'], model=fine_tuned_model)
    else: # 初回
        ft = openai.FineTune.create(training_file=uploaded_file['id'], model=BASE_MODEL)

    tuning_history.fine_tuning_job_id = ft['id']
# model = fine_tuned_model if fine_tuned_model else BASE_MODEL
# ft = openai.FineTune.create(training_file=uploaded_file['id'], model=model)
# # print(openai.FineTune.list())
# # print(ft['id'])

# ft = openai.FineTune.retrieve(id="ft-vUK64ddIb5bwobHKILYMN3VU")
# print(ft.model, ft.status, ft.fine_tuned_model)
# file_id = file['id']
# print(f'fileID: {file_id}')


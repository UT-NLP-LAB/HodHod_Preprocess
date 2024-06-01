
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
base_model_id = 'FacebookAI/xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True,
    use_fast =True,
    padding=False,
    truncation=False,
)
tokenizer.pad_token = tokenizer.eos_token
def xlm_token_counter(text):
    return len(tokenizer.tokenize(text))
def token_ratio_quality_assesment(text,filter_th=3):
    tokens = xlm_token_counter(text)
    text_len = len(text)
    if text_len/tokens >= filter_th:
        return True
    else:
        return False
def documents_filter_token_ratio(df,filter_th=3):
    # df = pd.read_json(f_path, lines=True)
    # df['tokens'] = df['text'].apply(xlm_token_counter)
    # df['token_ratio'] = [len(y)/x for (x,y) in zip (df['tokens'],df['text'])]
    # df = df[df.token_ratio>=filter_th]
    df = df[df['text'].apply(lambda x : token_ratio_quality_assesment(x,filter_th))]
    return df
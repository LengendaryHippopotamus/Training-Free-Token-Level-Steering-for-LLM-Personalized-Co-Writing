
def prompt_log(prompt_mode,dataset_name):
    if prompt_mode=="origin":
        prompt_log={
            "CDN":"I am writing a piece of code. I have completed part of it, and I would like you to continue from where I left off.\n\n"
        }

    elif prompt_mode in ["RAG","RAGS"]:
        prompt_log={
            "CDN":["I am a programmer, and here are some code (excerpts) I have written in the past.\n\n",
                       "\n\nI am currently writing a piece of code. I have completed part of it, and I would like you to continue from where I left off.\n\n"]
        }
    if prompt_mode=="chat":
        prompt_log={
            "CDN":"I am writing a piece of code. I have completed part of it, and I would like you to continue from where I left off. Please output your continuation directly without adding any other text.\n\n"
        }


    return prompt_log[dataset_name]

def language_log(dataset_name):
    language={
        "CDN":"python"
    }
    return language[dataset_name]



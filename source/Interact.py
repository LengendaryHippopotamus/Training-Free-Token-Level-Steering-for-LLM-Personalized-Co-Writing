import tkinter as tk
from tkinter import scrolledtext

def text_editor(initial_text):
    """带编辑功能的文本编辑器"""
    def save_text():
        nonlocal modified_text
        modified_text=text_area.get("1.0",tk.END).rstrip('\n')
        root.destroy()

    modified_text=initial_text  # 默认保留原文本

    root=tk.Tk()
    root.title("文本编辑器")
    text_area=scrolledtext.ScrolledText(root,wrap=tk.WORD,width=80,height=20)
    text_area.pack(padx=10,pady=10)
    text_area.insert(tk.INSERT,initial_text)

    tk.Button(root,text="保存并继续",command=save_text).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW",save_text)  # 窗口关闭时自动保存
    root.mainloop()

    return modified_text

def user_interaction(
    current_text,
    candidates
):
    # 用户选择
    while True:
        try:
            choice=int(input("\n请选择选项（-1结束）: "))
            if 1<=choice<=5 or choice==-1:
                break
        except:
            pass
        print("请输入有效选项！")

    if choice==-1:
        # break
        return -1

    # 拼接文本并编辑
    combined_text=current_text+candidates[choice-1]
    return combined_text
    # current_text=text_editor(combined_text)
    # return current_text
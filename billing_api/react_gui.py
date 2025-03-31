from reactpy import component, html, run
import stateful_optimization as rag
from utils import audio_master as am
from dotenv import load_dotenv
import string
import sys
#api keys and sources
audio_filename = "test_recording.m4a"
index_name = "billing-codes"
pc_key  = 'pcsk_321xz6_GZ7GkfHasMgmgw2Rc13HEezXzKBJqS13VEZGevovTzDhT4WLN21eYHrvPBovk9c'
@component
#testing controls 
def button_controller(display_text, message_text):
    """
    user interface method
    """
    
    def handle_event(event):

        match display_text:
            case "record":
                text =  am.audio_processing(audio_filename)
                f = open("target.txt", "a")
                f.write(text)
                f.close()
                return message_text
            case "generate codes":
                f = open('target.txt', 'r')
                content = f.read()
                f.close()
                result = rag.search_with_rag(content)
                print(result)
    return html.button({"on_click": handle_event}, display_text)

@component
def App():
    return html.div(
        button_controller("record", "listening"),
        button_controller("generate codes", "generating"),
    )

run(App)
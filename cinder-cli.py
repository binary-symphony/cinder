# ==================================================================================================
# IMPORTS
# ==================================================================================================


import subprocess
import os
import time
import keyboard
import argparse
import json
from pathlib import Path

from llama_cpp import Llama
from llama_cpp import llama_backend_init

from colorama import Fore, Back, Style, init
init(autoreset=True)


# ==================================================================================================
# CORE VARIABLES
# ==================================================================================================


VERSION = 0.12
VERBOSE_LOGGING = True
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models"
MODEL_METADATA_PATH = MODEL_PATH / "metadata"
MODEL_REGISTER_PATH = MODEL_METADATA_PATH / "registered_models.json"
STATUS_MSG_HITSTOP = .05


# ==================================================================================================
# UTILITY FUNCTIONS
# ==================================================================================================


def clear_screen():
    if not VERBOSE_LOGGING:
        subprocess.call("cls", shell=True)


# ==================================================================================================
# LOG RENDERING
# ==================================================================================================


def render_log(tag, style, origin_tag, origin_style, text, items=None):
    print(f"{origin_style}{origin_tag}{Style.RESET_ALL} - {style}[{tag}]: {text}{Style.RESET_ALL}")

def render_list(tag, style, origin_tag, origin_style, text, items=None):
    items = items or []
    list_header = f"{origin_style}{origin_tag}{Style.RESET_ALL} - {style}[{tag}]: {text}{Style.RESET_ALL}"
    list_contents = ""
    for item in items:
        list_contents+="|- " + str(item) + "\n"
    print(f"{list_header}\n{list_contents}")
    time.sleep(STATUS_MSG_HITSTOP)

def render_stream(tag, style, origin_tag, origin_style, text, items=None):
    total = ""
    print(f"{origin_style}{origin_tag}{Style.RESET_ALL} - {style}[{tag}]: {Style.RESET_ALL}")
    for chunk in text:
        text = chunk["choices"][0]["delta"].get("content", "")
        print(text, end="", flush=True)
        total = total + str(text)
    print("\n")
    time.sleep(STATUS_MSG_HITSTOP)
    return total

LEVELS = {
    "INF": ("INFO", Fore.YELLOW + Style.BRIGHT),
    "OK": ("OK", Fore.GREEN + Style.BRIGHT),
    "ERR": ("ERROR", Fore.RED + Style.BRIGHT),
    "MDL_OUT": ("MODEL_OUT", Fore.YELLOW + Style.BRIGHT)
}
ORIGINS = {
    "MAIN": ("<main>", ""),
    "CND": ("<cinder>", Fore.RED),
    "MDR": ("<model-register>", Fore.MAGENTA),
    "MDL": ("<model>", Fore.YELLOW)
}
TYPES = {
    "LOG": render_log,
    "LIST": render_list,
    "STREAM": render_stream
}

def format_log(text: str, level: str, log_type: str = "LOG", items:list=[], origin:str = "main"):
    tag, tag_style = LEVELS.get(level, ("[???]",""))
    origin_tag, origin_style = ORIGINS.get(origin, ("",""))
    renderer = TYPES.get(log_type.upper(), render_log)
    
    renderer(tag,tag_style,origin_tag,origin_style,text,items)

def log(text: str, level: str, log_type: str = "LOG", items: list=[], origin:str = "MAIN"):
    if VERBOSE_LOGGING or origin == "CND" or origin == "MDL":
        format_log(text,level,log_type,items,origin)
        time.sleep(STATUS_MSG_HITSTOP)


# ==================================================================================================
# CORE LOGIC
# ==================================================================================================


class Model:
    def __init__(self, name, quant, path):
        self.name = name
        self.quant = quant
        self.path = Path(path)
        self._llm = None
        
        self.history = []
        
    def __str__(self):
        return self.name + "|" + self.quant + "|" + str(self.path)
    
    def load(self):
        if self._llm is None:
            log(f"loading {self.name} into memory...","INF")
            self._llm = Llama(
                model_path=str(self.path),
                verbose = VERBOSE_LOGGING,
                n_gpu_layers=-1
            )
    
    def complete(self,prompt,tokens):
        self.load()
        response = self._llm.create_completion( #type: ignore
            prompt,
            repeat_penalty=1.1,
            max_tokens=tokens,
            stream=False
        )
        return response["choices"][0]["text"] #type: ignore
    
    def chat(self, system = None, token_max = 256):
        self.load()
        
        if system != None:
            self.history.append({
                "role":"system",
                "content": system
            })
        clear_screen()
        while True:
            user_input = input(f"{Fore.CYAN + Style.BRIGHT}<user>{Style.RESET_ALL} - {Fore.CYAN}[INPUT]:{Style.RESET_ALL} ")
            self.history.append({
                "role":"user",
                "content": user_input
            })
            response = log(self._llm.create_chat_completion(messages=self.history,max_tokens=token_max,stream=True),"MDL_OUT","STREAM",[],"MDL") #type: ignore
            self.history.append({
                "role":"assistant",
                "content":response
            })
            

class ModelRegister:
    def __init__(self):
        self.loaded_models = []
        
    def log(self, text: str, level: str, type: str = "LOG", items: list=[]):
        log(text,level,type,items,"MDR")
    
    def auto_register_models(self):
        with open (MODEL_REGISTER_PATH) as file:
            model_register = json.load(file)
            for model_info in model_register["models"]:
                model_path = (MODEL_PATH / model_info["path"].lstrip("/")).resolve()
                new_model = Model(model_info["name"],model_info["quant"],model_path)
                self.loaded_models.append(new_model)
            for model in self.loaded_models:
                with open(MODEL_METADATA_PATH / model.name,"a") as file:
                    pass
        self.log("models auto-registered into loaded_models","OK")
        self.log(f"model register found {len(self.loaded_models)} models", "INF")
        
    
    def search_models(self, args): # returns list of models in loaded models that contain keyword
        model_list = []
        for model in self.loaded_models:
            if args.lower() in str(model.name).lower():
                model_list.append(model)
        return model_list

    def find_model(self, args): # returns highest-priority model containing keyword
        model_list = self.search_models(args)
        if not model_list:
            return None
        return model_list[0]
    
    def list_models(self):
        clear_screen()
        log("Loaded Models","INF","LIST",self.loaded_models,"CND")

class CinderRuntime:
    def __init__(self):
        self.mr = ModelRegister()
    
    def log(self, text: str, level: str, type: str = "LOG", items: list=[]):
        log(text,level,type,items,"CND")
    
    def cmd_list_models(self,args):
        (self.mr).list_models()
        
    def cmd_complete(self, args):
        model = self.mr.find_model(args.model)
        if model is None:
            self.log("model not found","ERR")
            return
        result = model.complete(args.prompt,args.max_tokens)
        self.log(result,"INF")
        
    def cmd_chat(self, args):
        model = self.mr.find_model(args.model)
        if model is None:
            self.log("model not found","ERR")
            return
        model.chat(args.system, token_max=args.max_tokens)


# ================================================================================================
# COMMAND PARSER
# ================================================================================================


def build_parser(runtime: CinderRuntime):
    cmd_parser = argparse.ArgumentParser(prog="cinder")
    cmd_subparsers = cmd_parser.add_subparsers(dest="command", required=True)
    
    list_parser = cmd_subparsers.add_parser("list")
    list_parser.set_defaults(func = runtime.cmd_list_models)
    
    complete_parser = cmd_subparsers.add_parser("complete")
    complete_parser.add_argument("model")
    complete_parser.add_argument("prompt")
    complete_parser.add_argument("--max-tokens",
                            type = int,
                            default = 64,
                            help = "maximum number of tokens to generate")
    complete_parser.set_defaults(func = runtime.cmd_complete)
    
    chat_parser = cmd_subparsers.add_parser("chat")
    chat_parser.add_argument("model")
    chat_parser.add_argument("--max-tokens",
                            type = int,
                            default = 256,
                            help = "maximum number of tokens to generate")
    chat_parser.add_argument("--system",
                             type = str,
                             default = None,
                             help = "system prompt given to LLM")
    chat_parser.set_defaults(func = runtime.cmd_chat)
    
    return cmd_parser


# ==================================================================================================
# MAIN
# ==================================================================================================


def main():
    llama_backend_init()
    cinder = CinderRuntime()
    cinder.log("init cinder","OK")
    mr = cinder.mr
    mr.log("init model-register","OK")
    mr.log(f"attempting auto-register models from {MODEL_REGISTER_PATH}", "INF")
    mr.auto_register_models()
    
    parser = build_parser(cinder)
    args = parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()
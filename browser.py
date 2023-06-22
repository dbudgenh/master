import webbrowser
import time,pyautogui

def main():
    search_image_on_google_in_new_tab('cats')
    search_image_on_google_in_new_tab('dogs')
    time.sleep(3)
    close_tab()
    time.sleep(3)
    close_tab()

def open_google_image_search(query):
    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    webbrowser.open(search_url)

def search_image_on_google_in_new_tab(query):
    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    webbrowser.open_new_tab(search_url)

def close_tab():
    pyautogui.hotkey('ctrl', 'w')

if __name__ == '__main__':
    main()
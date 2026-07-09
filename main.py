from src.file_loader import *

def main():
    file_path = get_image_path()
    print(file_path)

    img = load_image(file_path)
    print(img.shape)


    
if __name__ == "__main__":
    main()
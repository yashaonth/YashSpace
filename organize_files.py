import os
import shutil

file_types = {
    'Images': ['.jpg', '.jpeg', '.png', '.gif'],
    'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx'],
    'Videos': ['.mp4', '.avi', '.mkv'],
    'Audio': ['.mp3', '.wav'],
    'Archives': ['.zip', '.rar'],
    'Scripts': ['.py', '.js', '.md'],
    'Config': ['.ini'],
    'Applications': ['.exe'],
    'Ltspice': ['.raw', '.asc', '.log'],
}


def ask_user(filename, ext):
    """Ask what to do with an unknown extension. Returns a folder name, or None to skip."""
    print(f"\n??? Unknown type '{ext or 'no extension'}': {filename}")
    print("1. Add to 'Miscellaneous'")
    print("2. Add to a NEW folder")
    print("3. Add to an EXISTING folder")
    print("4. Do not include in any folder")

    while True:
        choice = input("Choose 1-4: ").strip()

        if choice == "1":
            return "Miscellaneous"

        elif choice == "2":
            name = input("New folder name: ").strip()
            if name and name not in (".", "..") and name == os.path.basename(name):
                file_types.setdefault(name, []).append(ext)
                return name
            print("Invalid folder name.")

        elif choice == "3":
            names = list(file_types)
            for i, name in enumerate(names, 1):
                print(f"  {i}. {name}")
            pick = input("Folder number: ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(names):
                file_types[names[int(pick) - 1]].append(ext)
                return names[int(pick) - 1]
            print("Invalid number.")

        elif choice == "4":
            return None

        else:
            print("Please enter 1, 2, 3 or 4.")


def organize_folder(folder_path):
    print(f">>> Organizing files in: {folder_path} >>>\n")
    answers = {}  # unknown extension -> chosen folder, so each is only asked once
    moved = 0

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        folder = next((f for f, exts in file_types.items() if ext in exts), None)

        if folder is None:
            if ext not in answers:
                answers[ext] = ask_user(filename, ext)
            folder = answers[ext]

        if folder is None:
            print(f"<<< Skipped: {filename} >>>")
            continue

        target = os.path.join(folder_path, folder)
        os.makedirs(target, exist_ok=True)
        shutil.move(file_path, os.path.join(target, filename))
        print(f">>> Moved: {filename} → {folder}")
        moved += 1

    print(f"\n<<< Done! Total files moved: {moved} >>>")


if __name__ == "__main__":
    organize_folder(input("Enter the Folder Path: "))

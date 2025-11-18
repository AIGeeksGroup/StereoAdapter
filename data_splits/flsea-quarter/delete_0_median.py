import os

def delete_files_from_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"❌ File not found: {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    deleted = []
    failed = []

    for line in lines:
        file_path = line.strip()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted.append(file_path)
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
                failed.append(file_path)
        else:
            print(f"⚠️ File does not exist: {file_path}")
            failed.append(file_path)

    print(f"\n✅ Deleted {len(deleted)} files.")
    print(f"❌ Failed to delete {len(failed)} files.")
    if failed:
        print("Some files could not be deleted.")

if __name__ == "__main__":
    median_0_txt_path = 'data_splits/flsea/median_0.txt'
    delete_files_from_list(median_0_txt_path)

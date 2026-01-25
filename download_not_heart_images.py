import os
import requests
import time

folder = "data/processed_slices_by_class/Not_Heart"
os.makedirs(folder, exist_ok=True)

num_images = 500

for i in range(1, num_images + 1):
    url = f"https://picsum.photos/128/128?random={i}"
    success = False
    attempts = 0
    while not success and attempts < 3:  # retry 3 times
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(folder, f"img{i}.jpg"), "wb") as f:
                    f.write(response.content)
                success = True
                if i % 50 == 0:
                    print(f"{i} images downloaded...")
        except Exception as e:
            attempts += 1
            print(f"Retry {attempts} for image {i} due to error: {e}")
            time.sleep(2)  # wait 2 seconds before retry

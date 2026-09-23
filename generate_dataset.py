import os
import random
from PIL import Image, ImageDraw
import numpy as np

def generate_image(is_ship):
    # Base sea/land color (blue-greenish)
    base_color = (
        random.randint(20, 80),   # R
        random.randint(50, 120),  # G
        random.randint(100, 180)  # B
    )
    
    # Create image with noise
    img_array = np.zeros((80, 80, 3), dtype=np.uint8)
    for i in range(80):
        for j in range(80):
            noise = random.randint(-15, 15)
            img_array[i, j] = [
                max(0, min(255, base_color[0] + noise)),
                max(0, min(255, base_color[1] + noise)),
                max(0, min(255, base_color[2] + noise))
            ]
            
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Draw clouds/waves (white/light blue lines)
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, 80)
        y1 = random.randint(0, 80)
        x2 = x1 + random.randint(-20, 20)
        y2 = y1 + random.randint(-20, 20)
        draw.line((x1, y1, x2, y2), fill=(200, 220, 230, 100), width=random.randint(1, 3))
    
    if is_ship:
        # Draw a ship (gray/white rectangle with a shadow/wake)
        ship_w = random.randint(4, 8)
        ship_h = random.randint(12, 22)
        
        # Random position
        x = random.randint(10, 70 - ship_w)
        y = random.randint(10, 70 - ship_h)
        
        # Ship color
        ship_color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        
        # We can also rotate the ship using Image rotation but let's just draw rectangles for simplicity,
        # or use polygon to make it angled. Let's do polygon for angle.
        angle = random.uniform(0, 2 * np.pi)
        
        cx, cy = x + ship_w/2, y + ship_h/2
        
        # 4 corners
        dx1, dy1 = ship_w/2, ship_h/2
        dx2, dy2 = -ship_w/2, ship_h/2
        dx3, dy3 = -ship_w/2, -ship_h/2
        dx4, dy4 = ship_w/2, -ship_h/2
        
        def rotate(dx, dy):
            return cx + dx * np.cos(angle) - dy * np.sin(angle), cy + dx * np.sin(angle) + dy * np.cos(angle)
            
        p1 = rotate(dx1, dy1)
        p2 = rotate(dx2, dy2)
        p3 = rotate(dx3, dy3)
        p4 = rotate(dx4, dy4)
        
        # Draw wake
        draw.polygon([rotate(dx3, dy3 - 5), rotate(dx4, dy4 - 5), rotate(0, -ship_h/2 - random.randint(10,20))], fill=(220, 230, 240, 150))
        
        # Draw ship
        draw.polygon([p1, p2, p3, p4], fill=ship_color)

    return img

def main():
    print("Generating synthetic satellite imagery dataset...")
    random.seed(42)
    np.random.seed(42)
    
    os.makedirs('dataset/train_val/ship', exist_ok=True)
    os.makedirs('dataset/train_val/no_ship', exist_ok=True)
    os.makedirs('dataset/blind_test/ship', exist_ok=True)
    os.makedirs('dataset/blind_test/no_ship', exist_ok=True)
    
    # Generate Train/Val
    print("Generating Train/Val...")
    for i in range(800):  # 800 ships
        img = generate_image(is_ship=True)
        img.save(f'dataset/train_val/ship/img_{i}.png')
        
    for i in range(2400):  # 2400 no ships
        img = generate_image(is_ship=False)
        img.save(f'dataset/train_val/no_ship/img_{i}.png')
        
    # Generate Blind Test
    print("Generating Blind Test...")
    for i in range(200):   # 200 ships
        img = generate_image(is_ship=True)
        img.save(f'dataset/blind_test/ship/img_{i}.png')
        
    for i in range(600):   # 600 no ships
        img = generate_image(is_ship=False)
        img.save(f'dataset/blind_test/no_ship/img_{i}.png')
        
    print("Synthetic dataset generated successfully!")

if __name__ == "__main__":
    main()

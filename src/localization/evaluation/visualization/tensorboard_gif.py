import io
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_images_from_tensorboard(logdir, tag, output_folder):
    """
    Extracts images from a TensorBoard event file for a given tag and saves them as PNGs.

    Args:
        logdir (str): Path to TensorBoard log directory (e.g., 'runs/example')
        tag (str): Image tag name (e.g., 'random_images')
        output_folder (str): Folder to save extracted frames
    """
    ea = EventAccumulator(logdir)
    ea.Reload()

    # Check available image tags
    if tag not in ea.Tags().get("images", []):
        raise ValueError(f"Tag '{tag}' not found in TensorBoard logs. Available tags: {ea.Tags()['images']}")

    # Extract and save images
    imgs = ea.Images(tag)
    for i, img_data in enumerate(imgs):
        img = Image.open(io.BytesIO(img_data.encoded_image_string))
        img.save(f"{output_folder}/frame_{i:03d}.png")
    
    print(f"✅ Saved {len(imgs)} images to '{output_folder}'.")


from PIL import Image, ImageDraw, ImageFont

def create_gif_with_steps(image_files, output_path, duration=500, last_duration=2000, loop=0):
    """
    Creates a GIF with 'Step X' text added to each image (decreasing order),
    and the last image stays longer.

    Args:
        image_files (list[str]): List of image file paths (in order)
        output_path (str): Path to save the GIF
        duration (int): Duration per frame in milliseconds (for all except last)
        last_duration (int): Duration for the last frame (ms)
        loop (int): Loop count (0 = infinite)
    """
    frames = []
    total = len(image_files)

    for idx, file in enumerate(image_files):
        # Open image
        img = Image.open(file).convert("RGBA")
        draw = ImageDraw.Draw(img)

        # Try to use a clean font; fallback to default if not found
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        # Step number (decreasing)
        step_num = total - idx
        text = f"Step   {step_num}"

        # Get text bounding box to calculate dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Draw text (bottom-left corner)
        draw.text((10, img.height - text_h - 10)*10, text, fill="white", font=font, stroke_width=2, stroke_fill="black")

        frames.append(img)

    # Create GIF: last frame has longer duration
    durations = [duration] * (total - 1) + [last_duration]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        disposal=2
    )

    print(f"🎞️ GIF with steps saved to '{output_path}' ({total} frames).")


# Step 1: Extract TensorBoard images
extract_images_from_tensorboard("/home/sharghif/localization/lightning_logs/version_1720", "2d/Plot/Transformed", "frames")

# Step 2: Create GIF from extracted images
import glob

image_files = sorted(glob.glob("frames/frame_*.png"))
create_gif_with_steps(image_files, "output_steps.gif", duration=400, last_duration=3000)


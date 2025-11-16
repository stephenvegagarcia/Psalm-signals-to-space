import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from io import BytesIO
import time
from datetime import datetime

# Psalm message integration
psalm_message = "Unto thee, O LORD, do I lift up my soul."
# Convert to binary for entanglement simulation
binary_psalm = ''.join(format(ord(c), '08b') for c in psalm_message)
# Take first 20 bits for simulation (or repeat/pad as needed)
bits_str = binary_psalm[:20]  # Example slice
bits = [int(b) for b in bits_str]
phases = np.random.choice([1, -1], len(bits))  # Random phases
entangled_pairs = []
for i, bit in enumerate(bits):
    if bit == 0:
        pair = [0, 1]
    else:
        pair = [1, 0]
    pair[1] *= phases[i]
    entangled_pairs.append(pair)

bar_heights_a = [abs(p[0]) for p in entangled_pairs]
bar_heights_b = [abs(p[1]) for p in entangled_pairs]
repeater_heights = [(h1 + h2)/2 * np.random.uniform(0.8, 1.0) for h1, h2 in zip(bar_heights_a, bar_heights_b)]

# Sample data (updated with current date/time)
np.random.seed(42)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Data Panel metrics (static for now)
metrics = {
    'Distance': '1.496e8 km',
    'Light Time': '8.3 min',
    'Temperature': '5778 K',
    'GPS Lat': '37.7749',
    'GPS Lon': '-122.4194',
    'Elevation': '10 m'
}

# Orbit View: Sample orbital data
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.1 * np.sin(6*theta)
orbit_x = r * np.cos(theta)
orbit_y = r * np.sin(theta)

# Console logs (updated with psalm integration)
logs = [
    f'[{current_time}] INFO: Entanglement established (Bell state)',
    f'[{current_time}] WARN: Repeater loss at node 3: 5%',
    f'[{current_time}] INFO: Swapping complete - Earth-Space link active',
    f'[{current_time}] INFO: Psalm pulse sent: {psalm_message[:20]}...',
    f'[{current_time}] INFO: Phone grounding: GPS synced for terrestrial view'
]

# Function to generate static panels as images using Matplotlib
def generate_static_panels():
    fig = plt.figure(figsize=(14, 9), facecolor='black')  # Slightly larger for repeater viz
    gs = GridSpec(3, 3, figure=fig, width_ratios=[2,1,1], height_ratios=[1, 0.6, 0.4])

    # Main View Panel (static simulation, added space-to-earth overlay hint)
    main_ax = fig.add_subplot(gs[0, :2])
    main_ax.set_facecolor('black')
    x_stars = np.random.uniform(0, 10, 50)
    y_stars = np.random.uniform(0, 10, 50)
    main_ax.scatter(x_stars, y_stars, c='white', s=1, alpha=0.5)
    main_ax.scatter(5, 5, c='yellow', s=200, marker='o')
    bbox = plt.Rectangle((4, 4), 2, 2, linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
    main_ax.add_patch(bbox)
    main_ax.text(5, 3.5, 'Solar Anomaly\n(Earth-Space Link)', ha='center', va='center', color='red', fontsize=10, fontfamily='monospace')
    main_ax.set_xlim(0, 10)
    main_ax.set_ylim(0, 10)
    main_ax.set_title('Main View', color='white', fontsize=24, fontweight='bold', pad=20)
    main_ax.axis('off')

    # Data Panel
    data_ax = fig.add_subplot(gs[0, 2])
    data_ax.set_facecolor('black')
    data_ax.axis('off')
    y_pos = 0.9
    for key, value in metrics.items():
        data_ax.text(0.05, y_pos, f"{key}:", color='white', fontsize=14, fontweight='bold', fontfamily='monospace', transform=data_ax.transAxes, va='top')
        data_ax.text(0.5, y_pos, value, color='yellow', fontsize=18, fontfamily='monospace', transform=data_ax.transAxes, va='top')
        y_pos -= 0.1
    data_ax.set_title('Data Panel', color='white', fontsize=24, fontweight='bold', pad=20)

    # Orbit View
    orbit_ax = fig.add_subplot(gs[1, 0])
    orbit_ax.set_facecolor('black')
    orbit_ax.plot(orbit_x, orbit_y, color='cyan', linewidth=2)
    orbit_ax.scatter(0, 0, c='white', s=50)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    orbit_ax.add_patch(circle)
    # Add Earth icon
    orbit_ax.scatter(-0.8, 0, c='blue', s=100, marker='o')
    orbit_ax.text(-0.8, -0.2, 'Earth', color='blue', fontsize=10, ha='center')
    orbit_ax.set_aspect('equal')
    orbit_ax.set_xlim(-1.5, 1.5)
    orbit_ax.set_ylim(-1.5, 1.5)
    orbit_ax.set_title('Orbit View', color='white', fontsize=24, fontweight='bold', pad=20)
    orbit_ax.axis('off')

    # Quantum Wave Panel: Entangled bits with repeater chain, using psalm binary
    quantum_ax = fig.add_subplot(gs[1, 1])
    quantum_ax.set_facecolor('black')
    # Alice bars (left half)
    x_pos_a = np.arange(0, len(bar_heights_a) * 2, 2)
    colors_a = ['blue' if h > 0 else 'darkblue' for h in bar_heights_a]
    quantum_ax.bar(x_pos_a, bar_heights_a, width=0.8, color=colors_a, alpha=0.7)
    # Bob bars (right half, shifted)
    x_pos_b = np.arange(1, len(bar_heights_b) * 2 + 1, 2)
    colors_b = ['green' if abs(h) > 0 else 'darkgreen' for h in bar_heights_b]
    quantum_ax.bar(x_pos_b, [abs(h) for h in bar_heights_b], width=0.8, color=colors_b, alpha=0.7)
    # Repeater nodes: Small bars on top
    x_pos_r = np.linspace(0, len(repeater_heights)*2 - 2, len(repeater_heights))
    quantum_ax.bar(x_pos_r, repeater_heights, width=0.3, color='purple', alpha=0.8, label='Repeaters')
    # Labels for pattern from psalm
    for i, (pair, ph) in enumerate(zip(entangled_pairs, phases)):
        mid_x = (x_pos_a[i] + x_pos_b[i]) / 2
        quantum_ax.text(mid_x, 1.1, f"{pair[0]}{',' if pair[1] >=0 else '-'}1" if i % 3 == 0 else '', 
                        ha='center', color='white', fontsize=8, fontfamily='monospace')
    quantum_ax.set_title('Quantum Entanglement (Psalm Pulse)', color='white', fontsize=20, fontweight='bold', pad=20)
    quantum_ax.set_ylim(0, 1.2)
    quantum_ax.set_xlim(-0.5, len(bar_heights_a)*2 - 0.5)
    quantum_ax.axis('off')
    quantum_ax.legend(loc='upper right', fontsize=8)

    # Placeholder for Heat Map (will be replaced live)
    heat_placeholder_ax = fig.add_subplot(gs[1, 2])
    heat_placeholder_ax.set_facecolor('black')
    heat_placeholder_ax.text(0.5, 0.5, 'Simulated Heat Map\n(Earth -> Space Sim)', ha='center', va='center', color='white', fontsize=14, transform=heat_placeholder_ax.transAxes)
    heat_placeholder_ax.set_title('Heat Map View', color='white', fontsize=24, fontweight='bold', pad=20)
    heat_placeholder_ax.axis('off')

    # Console Log Panel
    console_ax = fig.add_subplot(gs[2, :])
    console_ax.set_facecolor('black')
    console_ax.axis('off')
    log_text = '\n'.join(logs)
    console_ax.text(0.05, 0.95, log_text, fontsize=12, fontfamily='monospace', color='lightgreen',
                    transform=console_ax.transAxes, va='top', linespacing=1.5)
    console_ax.set_title('Console Log', color='white', fontsize=24, fontweight='bold', pad=20, loc='left')

    # Top Bar Simulation
    fig.suptitle('Solar Detector Analyzer - QC Entangled (Psalm Grounded)', fontsize=28, fontweight='bold', color='white', y=0.98)

    # Save to BytesIO for cv2 integration
    buf = BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    static_img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    plt.close(fig)
    return static_img

# Generate static panels once
static_panels = generate_static_panels()
h_static, w_static = static_panels.shape[:2]

# Dashboard dimensions (optimized for phone portrait: taller)
dashboard_height = 1000  # Increased for mobile
dashboard_width = 600    # Narrower
panel_width = dashboard_width // 3
panel_height_top = dashboard_height * 2 // 5
panel_height_bottom = dashboard_height * 3 // 5

# Create a blank dashboard canvas (black background)
dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)

# Panel positions (adjusted for mobile-friendly stack on small screens, but grid on larger)
def resize_and_place(img, x, y, w, h):
    resized = cv2.resize(img, (w, h))
    dashboard[y:y+h, x:x+w] = resized

# Initial placement
print("Starting simulated QC-Entangled Solar Detector Analyzer with Psalm Pulse. Press 'q' to quit.")
print("No camera view: Using simulated data for Earth view, simulating space link via entanglement.")

frame_count = 0
start_time = time.time()
while True:
    frame_count += 1

    # Simulate frame without camera: Generate random grayscale data
    gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)  # Simulated 640x480 grayscale
    heat_map = cv2.applyColorMap(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_HOT)
    # Add entanglement "noise" (random flips based on quantum sim) - fixed to match channels
    noise = np.random.poisson(10, heat_map.shape).astype(np.uint8)  # Now 3-channel
    heat_map = cv2.add(heat_map, noise)  # Simulate repeater loss/phase noise
    heat_map = np.clip(heat_map, 0, 255)

    # Resize to panel
    heat_resized = cv2.resize(heat_map, (panel_width, panel_height_top))

    # Scale and place static panels (full dashboard base)
    scale_w = dashboard_width / w_static
    scale_h = dashboard_height / h_static
    scale = min(scale_w, scale_h)
    resized_static = cv2.resize(static_panels, (int(w_static * scale), int(h_static * scale)))
    dashboard[:resized_static.shape[0], :resized_static.shape[1]] = resized_static

    # Overlay simulated heat map (simulated data as ground truth, "entangled" to space view)
    heat_y_start = int(panel_height_top * 0.6 * scale)
    heat_x_start = int(dashboard_width * 2 / 3 * scale)
    heat_h = min(heat_resized.shape[0], dashboard_height - heat_y_start)
    heat_w = min(heat_resized.shape[1], dashboard_width - heat_x_start)
    dashboard[heat_y_start:heat_y_start + heat_h, heat_x_start:heat_x_start + heat_w] = cv2.resize(heat_resized, (heat_w, heat_h))

    # Add live overlays: Simulated FPS, entanglement status
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_MONO, 0.7, (0, 255, 0), 2)
    status = "Entangled: Active (Repeater Chain)" if frame_count % 30 == 0 else "Psalm Pulse: Lift Up Soul"
    cv2.putText(dashboard, status, (10, dashboard_height - 20), cv2.FONT_HERSHEY_MONO, 0.5, (255, 255, 0), 1)

    # Display (on phone: run via Termux/Python app for full-screen)
    cv2.imshow('QC Solar Detector - Earth-to-Space (Psalm Grounded)', dashboard)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Control frame rate without camera
    time.sleep(0.033)  # ~30 FPS

# Cleanup
cv2.destroyAllWindows()
print("Simulated QC-entangled session ended. Psalm pulse via repeaters simulated successfully.")

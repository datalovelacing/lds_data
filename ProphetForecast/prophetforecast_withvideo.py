#%%
import pandas as pd
import numpy as np
import time

import plotly.express as px
#import plotly.graph_objects as go

import os
import cv2 # pip install opencv-python
import glob
from datetime import datetime

#%% --- 1. Load and Prepare Data ---
df = pd.read_csv('LivingApostles.csv')
df['Born'] = pd.to_datetime(df['Born'])

# Calculate Age (robust to different NumPy versions)
today = pd.Timestamp('today')
df['Age'] = ((today - df['Born']).dt.days / 365.25).astype(int)

# Sort by Seniority (1 = Current Prophet)
df = df.sort_values('Seniority').reset_index(drop=True)
YEAR_START = today.year

# Load Death Probs (2019 SSA Data)
deathm_df = pd.read_excel("DeathProbsE_M_Hist_TR2024.xlsx", header=1).set_index('Year')
death_probs_array = deathm_df.query('Year==2019').T.iloc[:, 0].values


# Ensure 100% mortality at age 115
if len(death_probs_array) > 115:
    death_probs_array[115:] = 1.0
    
#%% --- 2. Simulation Functions ---

def single_simulation(names, initial_ages, death_probs, years_outlook, rands, iteration_id):
    """
    Runs one single timeline (iteration) of the simulation.
    """
    num_people = len(names)
    is_living = np.ones(num_people, dtype=bool)
    current_ages = initial_ages.copy()
    
    # Store results for this specific timeline
    history = []
    
    # Year 0: Current status (The most senior is Prophet at 100% prob)
    history.append((iteration_id, 0, names[0]))
    
    for y in range(1, years_outlook + 1):
        # 1. Mortality Lookup
        lookup_ages = np.clip(current_ages, 0, len(death_probs) - 1)
        p_death = death_probs[lookup_ages]
        
        # 2. Survival Check (using slice of pre-generated randoms)
        is_living &= (rands[y-1, :] >= p_death)
        
        # 3. Identify Prophet
        living_indices = np.where(is_living)[0]
        prophet = names[living_indices[0]] if living_indices.size > 0 else "Unknown"
        
        history.append((iteration_id, y, prophet))
        
        # 4. Aging
        current_ages += 1
        
    return history

def run_monte_carlo(df, death_probs, years_outlook, n_iter):
    """
    Manages the loop and aggregates results from multiple simulations.
    """
    names = df['Name'].values
    initial_ages = df['Age'].values
    num_people = len(names)
    
    # Pre-generate all random numbers for all iterations at once
    all_rands = np.random.random((n_iter, years_outlook, num_people))
    
    all_results = []
    
    for i in range(n_iter):
        # Pass the pre-generated randoms for JUST this iteration
        iter_res = single_simulation(
            names, 
            initial_ages, 
            death_probs, 
            years_outlook, 
            all_rands[i], 
            iteration_id=i
        )
        all_results.extend(iter_res)
        
    return pd.DataFrame(all_results, columns=['iter', 'YearOffset', 'Prophet'])

#%% --- 3. Execution ---
N_ITER = 10000
YEARS_OUTLOOK = 50

start_time = time.time()
sim_results = run_monte_carlo(df, death_probs_array, YEARS_OUTLOOK, N_ITER)
print(f"Total Simulation took {time.time() - start_time:.2f} seconds")

#%% --- 4. Post-Processing ---
sim_results['Year'] = sim_results['YearOffset'] + YEAR_START

# Yearly Probabilities (Pivot table)
yearly_pivot = (
    sim_results.groupby(['Year', 'Prophet'])
    .size()
    .unstack(fill_value=0) / N_ITER
)

# Create the long-format probability table
all_iter_grp = sim_results.groupby(['Year', 'Prophet']).size().reset_index(name='Obs')
all_iter_grp['Prob'] = all_iter_grp['Obs'] / N_ITER

#%% Calculate Summary Stats
prob_ever = sim_results.groupby('Prophet')['iter'].nunique() / N_ITER
total_years_count = sim_results.groupby('Prophet').size()
avg_service = total_years_count / (prob_ever * N_ITER)

summary_df = pd.DataFrame({
    'PBecomingProphet': prob_ever,
    'YearsServiceIfProphet': avg_service
}).reset_index()

summary_df['ExpectedYearsService'] = summary_df['PBecomingProphet'] * summary_df['YearsServiceIfProphet']
final_summary = summary_df.merge(df[['Name', 'Age', 'Seniority']], left_on='Prophet', right_on='Name').drop(columns='Name')

print("\n--- Summary Results ---")
print(final_summary.sort_values('Seniority'))

# Export
final_summary.to_csv('prophet_sim_summary.csv', index=False)
all_iter_grp.to_csv('prophet_yearly_probabilities.csv')

#%% Plot final probs
# 1. Prepare and Sort the Final Summary Data
# We sort ascending=True so the highest probability is at the top when plotted on 'h' bars
final_summary = final_summary.sort_values('Seniority', ascending=False)

# 2. Create the Static Summary Chart
fig_static = px.bar(final_summary, 
             x="PBecomingProphet", 
             y="Prophet", 
             color="PBecomingProphet",
             text="PBecomingProphet",
             orientation='h',
             #color_continuous_scale='Bluyl',
             color_continuous_scale='Sunset_r',
             template="plotly_white")

# 3. Final Layout - Matching your animation style exactly
fig_static.update_layout(
    width=400,
    height=600,
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=0, r=40, t=70, b=50), 
    
    # Global font setting
    font=dict(family="Inter, sans-serif"),
    
    # Matching your "Fake Title" and Subtitle structure
    annotations=[
        # 1. THE "FAKE" TITLE
        dict(
            text="<b>Prophet Forecast (as of March 2026)</b>", 
            xref="paper", yref="paper",
            x=0.2, y=1.12,
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor='center'
        ),
        # 2. THE SUBTITLE
        dict(
            text="<b>Probability of Becoming LDS Prophet",
            xref="paper", yref="paper",
            x=0.2, y=1.06,
            showarrow=False,
            font=dict(size=14, color="#3b3a3a"),
            xanchor='center'
        )
    ],
    
    yaxis={
        'tickfont': {'color': '#3b3a3a', 'size': 14},
        'title': '',
           },
    #xaxis=dict(visible=False),
    xaxis={
        #'visible':False,
        'title': '',
        'tickvals': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'tickformat': '.0%',
        'range': [0, 0.8],
        'tickfont': {'color': '#3b3a3a'},
        'showgrid': True,
        'showticklabels':False,
    }
)

# 4. Update Bar Text for percentages
fig_static.update_traces(
    texttemplate='%{text:.1%}', 
    textposition='outside',
    textfont=dict(family="Inter, sans-serif", color='black'),
    cliponaxis=False,
)

# Create the directory if it doesn't exist
if not os.path.exists('frames'):
    os.makedirs('frames')

# 5. Save for the Video Stitching
fig_static.write_image("frames/summary_final.png", width=400, height=600, scale=5)
fig_static.show(renderer="browser")

#%% Video Chart Prep

# Prepare data (ensure "Unknown" is part of the possible Prophets)
years = all_iter_grp['Year'].unique()
# Add "Unknown" to the list of names to ensure it's tracked in the loop
prophets = list(all_iter_grp['Prophet'].unique())
if "Unknown" not in prophets:
    prophets.append("Unknown")

full_index = pd.MultiIndex.from_product([years, prophets], names=['Year', 'Prophet'])
animated_df_full = all_iter_grp.set_index(['Year', 'Prophet']
                              ).reindex(full_index, fill_value=0).reset_index()

# Re-merge Seniority
# We use a left merge so "Unknown" stays in the dataframe but gets a NaN Seniority
animated_df_full = animated_df_full.merge(df[['Name', 'Seniority']], 
                                          left_on='Prophet', right_on='Name', how='left')

# Give "Unknown" a seniority value that puts it at the very bottom of the chart
animated_df_full.loc[animated_df_full['Prophet'] == 'Unknown', 'Seniority'] = 99

# Reverse seniority: Prophet (1) at Top, Unknown (99) at the very Bottom
y_order = animated_df_full[['Prophet', 'Seniority']].drop_duplicates(
    ).sort_values('Seniority', ascending=True)['Prophet'].tolist()

#%% Interactive Chart

# Create the Chart
fig = px.bar(animated_df_full, 
             x="Prob", 
             y="Prophet", 
             color="Prophet",
             orientation='h',
             #text="Prob",
             animation_frame="Year", 
             animation_group="Prophet",
             range_x=[0, 0.5],
             #title=f"LDS Prophet Simulation {YEAR_START}",
             labels={"Prob": "Probability"},
             category_orders={"Prophet": y_order}, 
             template="plotly_white")

# Final Layout with "Double Annotation" for Title/Subtitle
fig.update_layout(
    showlegend=False,
    margin=dict(l=0, r=40, t=70, b=50), 
    width=400,
    height=600,
    
    # Global font setting for the whole figure
    font=dict(family="Inter, sans-serif"),
    
    # We leave the official title empty so it doesn't interfere
    title={'text': ""}, 
    
    annotations=[
        # 1. THE "FAKE" TITLE
        dict(
            text="<b>Who will be the prophet in </b>", # Bolded for title look
            xref="paper", 
            yref="paper",
            x=0,       # Aligns it with the Prophet names on the far left
            y=1.12,        # Higher up
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor='left'
        ),
        # 2. THE SUBTITLE
        dict(
            text=f"Probability for Year: <b>{YEAR_START}<b>",
            xref="paper", 
            yref="paper",
            x=0,       # EXACT same x as the title above
            y=1.06,        # Slightly lower than title
            showarrow=False,
            font=dict(size=14, color="#3b3a3a"),
            xanchor='left'
        )
    ],
    
    yaxis={'title': '', 
           'tickfont': {'color': '#3b3a3a', 'size': 14},},
    #xaxis={'title': ''}
    xaxis={
        'title': 'Probability',
        'tickvals': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'tickformat': '.0%',  # Rounds to 0 decimal places and adds %
        'range': [0, 0.6]    # Fixed range to prevent the chart from jumping
    },
)

# Update the ""Fake Title for every frame in the animation
for frame in fig.frames:
    year_val = frame.name
    # Note: frame.layout.annotations[0] is the title we just created
    frame.layout.annotations = [
        # 1. THE "FAKE" TITLE
        dict(
            text="<b>LDS Prophet Forecast</b>", # Bolded for title look
            xref="paper", 
            yref="paper",
            x=0.2,       # Aligns it with the Prophet names on the far left
            y=1.12,        # Higher up
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor='left'
        ),
        # 2. THE SUBTITLE
        dict(
            text=f"Probability for Year: <b>{year_val}</b>",
            xref="paper", 
            yref="paper",
            x=0.2,       # EXACT same x as the title above
            y=1.06,        # Slightly lower than title
            showarrow=False,
            font=dict(size=14, color="#3b3a3a"),
            xanchor='left'
        )
    ]

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 250

fig.show(renderer="browser")

#%% Video Chart for Video

# --- 1. Create Smooth Interpolated Data ---
frames_per_year = 3  # 10 steps between each year
interpolated_data = []

# Sort years to ensure we move chronologically
years_sorted = sorted(animated_df_full['Year'].unique())

for i in range(len(years_sorted) - 1):
    year_start = years_sorted[i]
    year_end = years_sorted[i+1]
    if year_start == 2060:
        continue
    
    df_start = animated_df_full[animated_df_full['Year'] == year_start].set_index('Prophet')
    df_end = animated_df_full[animated_df_full['Year'] == year_end].set_index('Prophet')
    
    for step in range(frames_per_year):
        alpha = step / frames_per_year
        # Linear interpolation of the Probability value
        df_step = df_start.copy()
        df_step['Prob'] = df_start['Prob'] * (1 - alpha) + df_end['Prob'] * alpha
        df_step['SubYear'] = year_start + alpha  # Unique ID for the frame
        interpolated_data.append(df_step.reset_index())

# Final dataframe for the animation
smooth_df = pd.concat(interpolated_data)

# --- 2. Create the Chart with the Smooth Data ---
fig = px.bar(smooth_df, 
             x="Prob", 
             y="Prophet", 
             color="Prophet",
             orientation='h',
             animation_frame="SubYear", # Use the fractional year
             range_x=[0, 0.6],
             labels={"Prob": "Probability"},
             category_orders={"Prophet": y_order}, 
             template="plotly_white")


#%% save frames for video

# Kill UI elements on the main figure
fig.layout.sliders = None
fig.layout.updatemenus = None
fig.layout.showlegend = False

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Smooth Frame Generation...")

for i, frame in enumerate(fig.frames):
    # Update the data for this specific sub-step
    fig.update(data=frame.data)
    
    current_year = int(np.floor(float(frame.name)))
    title_text = f"Who will be the prophet in <b>{current_year}</b>?"
    
    if current_year >= 2065:
        continue

    # 3. APPLY FORMATTING HERE (Every frame needs this)
    fig.update_layout(
        #annotations=frame.layout.annotations,
        annotations=[
            dict(
                text=title_text,
                xref="paper",
                yref="paper",
                x=0.2,        # Horizontally centered
                y=1.07,        # Positioned slightly above the plot
                showarrow=False,
                font=dict(size=20, color="#3b3a3a"),
                xanchor="center"
            )
        ],
        yaxis={'title': ''},  # Remove "Prophet" title from Y-axis
        xaxis={
            'title': 'Probability',
            'tickvals': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'tickformat': '.0%',  # Adds the % sign back
            'range': [0, 0.6]     # Keeps the scale from jumping
        },
        #margin=dict(l=180, r=40, t=100, b=50)
        margin=dict(l=0, r=40, t=50, b=50), 
    )

    # Save the PNG
    file_name = f"frames/frame_{i:03d}.png"
    fig.write_image(file_name, width=400, height=600, scale=2)

    #if i % 20 == 0 or i == len(fig.frames) - 1:
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] Rendered {file_name} ({i+1}/{len(fig.frames)})")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Done! {len(fig.frames)} clean frames ready.")
        
        
#%% New slide

import os
from PIL import Image, ImageDraw, ImageFont

def create_intro_slide(
    text, 
    filename="intro_text.png", 
    width=400, 
    height=600, 
    background_color="white", 
    text_color="#1a1a1a", 
    font_path="georgiab.ttf", 
    font_size=32
):
    # 1. Setup Folder
    folder = "frames"
    if not os.path.exists(folder): os.makedirs(folder)
    save_path = os.path.join(folder, filename)

    # 2. Create Canvas
    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # 3. Load Font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 4. HANDLE MANUAL LINE BREAKS + DYNAMIC WRAPPING
    max_pixel_width = width * 0.75 
    final_lines = []
    
    input_paragraphs = text.strip().split('\n')

    for paragraph in input_paragraphs:
        words = paragraph.split()
        
        # Handle empty lines/manual breaks
        if not words:
            final_lines.append("")
            continue
            
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_pixel_width:
                current_line.append(word)
            else:
                final_lines.append(' '.join(current_line))
                current_line = [word]
        
        # FIX: Append the remainder of the words
        if current_line:
            final_lines.append(' '.join(current_line))

    # 5. VERTICAL CENTERING (ROBUST CALCULATION)
    sample_bbox = draw.textbbox((0, 0), "Hg", font=font)
    line_h = sample_bbox[3] - sample_bbox[1]
    standard_spacing = 16 
    empty_line_gap = 16  # This is your "tiny" space

    total_text_height = 0
    for i, line in enumerate(final_lines):
        if line.strip():  # If line has text
            total_text_height += line_h
            # Add spacing if there's another line coming (and current isn't last)
            if i < len(final_lines) - 1:
                total_text_height += standard_spacing
        else:  # If line is empty
            total_text_height += empty_line_gap

    current_y = (height - total_text_height) // 2 - 14

    # 6. DRAW
    for line in final_lines:
        if line.strip():
            # Draw the text
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            draw.text(((width - line_w) // 2, current_y), line, font=font, fill=text_color)
            
            # Move down by line height + standard spacing
            current_y += line_h + standard_spacing
        else:
            # Move down by ONLY the tiny gap
            current_y += empty_line_gap

    img.save(save_path, "PNG", quality=95)
    print(f"Success: {save_path} generated.")
    return save_path

# Test
text = """
Latter-day Saint Prophet Forecast

Who's next?

10,000 possible paths for the future reveal...


  
"""

create_intro_slide(
    text=text,
    text_color="#413a66",
    font_path="segoeuib.ttf", # Ensure this font is installed on your system!
    font_size=33
)

#%% new video render

# --- 1. Helper Functions ---

def format_for_reel_with_border(img, bg_image, target_w=1080, target_h=1920):
    """Scales image to fit with margins on ALL sides for the 'floating card' look."""
    if img is None: return None
    h, w = img.shape[:2]
    
    # 92% width and 75% height to stay safe for 3:4 and 9:16 crops
    width_limit = target_w * 0.92 
    height_limit = target_h * 0.75 
    
    scale = min(width_limit / w, height_limit / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = bg_image.copy()
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def add_branding(frame):
    """Draws the Hook and Handle on the final frame to prevent pixel-shifting."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # TOP HOOK (Positioned for 3:4 Grid Safety)
    hook_text = "FUTURE PROPHET PREDICTIONS"
    hook_size = cv2.getTextSize(hook_text, font, 1.4, 3)[0]
    hx, hy = (1080 - hook_size[0]) // 2, 200 # Moved down slightly for better 3:4 framing
    
    # Shadow + Text
    cv2.putText(frame, hook_text, (hx+3, hy+3), font, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hook_text, (hx, hy), font, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    
    # BOTTOM HANDLE (@lds_data)
    handle_text = "@lds_data"
    handle_size = cv2.getTextSize(handle_text, font, 1.0, 2)[0]
    bx, by = (1080 - handle_size[0]) // 2, 1750 # Safe above the Reel UI
    
    # Shadow + Text
    cv2.putText(frame, handle_text, (bx+2, by+2), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, handle_text, (bx, by), font, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    
    return frame

def ease_in_out_cubic(t):
    return 4 * t**3 if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

# --- 2. Main Render Logic ---

# File Paths
frame_files = sorted(glob.glob('frames/frame_*.png'))
intro_path = 'frames/intro_text.png'
summary_path = 'frames/summary_final.png'
bg_path = 'purple.png'

# Verification
if not frame_files or not os.path.exists(summary_path):
    print("!!! Error: Make sure your 'frames' folder and 'summary_final.png' exist.")
    exit()

# Load Background
raw_bg = cv2.imread(bg_path)
if raw_bg is None:
    raw_bg = cv2.imread('purple.jpg') # Automatic fallback

if raw_bg is None:
    print("!!! Error: purple.png not found in this folder.")
    exit()

# Setup Video Writer
REEL_WIDTH, REEL_HEIGHT = 1080, 1920
bg_resized = cv2.resize(raw_bg, (REEL_WIDTH, REEL_HEIGHT))

# USING XVID FOR WINDOWS STABILITY
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('lds_prophets_reel.mp4', fourcc, 30, (REEL_WIDTH, REEL_HEIGHT))

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Render...")

# Pre-format the base layers (No text yet to ensure perfect fades)
s_base1 = format_for_reel_with_border(cv2.imread(summary_path), bg_resized)
s_base2 = format_for_reel_with_border(cv2.imread(intro_path), bg_resized)
first_base = format_for_reel_with_border(cv2.imread(frame_files[0]), bg_resized)
last_base = format_for_reel_with_border(cv2.imread(frame_files[-1]), bg_resized)

# STEP A: Initial Summary Hook (1 Second)
for _ in range(30):
    out.write(add_branding(s_base1.copy()))
    
# STEP B: Fade from Summary to Intro (15 frames)
for t in np.linspace(0, 1, 15):
    alpha = ease_in_out_cubic(t)
    blended = cv2.addWeighted(s_base1, 1 - alpha, s_base2, alpha, 0)
    out.write(add_branding(blended))

# STEP C: Hold on Intro (2 Seconds)
intro_frame = add_branding(s_base2.copy())
for _ in range(30 * 3):
    out.write(intro_frame)

# STEP D: Fade from Intro to Animation (15 frames)
for t in np.linspace(0, 1, 15):
    alpha = ease_in_out_cubic(t)
    blended = cv2.addWeighted(s_base2, 1 - alpha, first_base, alpha, 0)
    out.write(add_branding(blended))

# STEP E: The Main Animation
for filename in frame_files:
    img = cv2.imread(filename)
    reel_base = format_for_reel_with_border(img, bg_resized)
    # Write each frame 1 times for a fast pace
    for _ in range(2): 
        out.write(add_branding(reel_base.copy()))

# STEP F: Fade back to Summary (15 frames)
for t in np.linspace(0, 1, 15):
    alpha = ease_in_out_cubic(t)
    blended = cv2.addWeighted(last_base, 1 - alpha, s_base1, alpha, 0)
    out.write(add_branding(blended))

# STEP G: Final Hold on Summary (4 Seconds)
final_frame = add_branding(s_base1.copy())
for _ in range(30 * 4):
    out.write(final_frame)

out.release()
print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: 'lds_prophets_reel.mp4' created.")


#%%
# --- 3. Post-Render Image Export ---
import os

# Create a dedicated folder for the individual stills
export_folder = 'exported_stills'
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Video closed. Exporting all formatted stills...")

# 1. Export the Special Frames (Intro & Summary)
# We use the pre-formatted bases (s_base1, s_base2) from Step 2
cv2.imwrite(os.path.join(export_folder, "00_summary_hook.png"), add_branding(s_base1.copy()))
cv2.imwrite(os.path.join(export_folder, "01_intro_text.png"), add_branding(s_base2.copy()))

# 2. Export the Main Animation Sequence
for i, filename in enumerate(frame_files):
    # Load and process the frame
    img = cv2.imread(filename)
    reel_base = format_for_reel_with_border(img, bg_resized)
    
    # Apply branding
    final_still = add_branding(reel_base)
    
    # Save with a prefix to keep them ordered after the intro
    save_name = os.path.join(export_folder, f"02_prophet_frame_{i:03d}.png")
    cv2.imwrite(save_name, final_still)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Done! All images saved to '{export_folder}'.")
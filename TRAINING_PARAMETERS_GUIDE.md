# คู่มือพารามิเตอร์การเทรน Humanoid-10102025 ด้วย RSL-RL

## สารบัญ
1. [พารามิเตอร์ Command Line](#พารามิเตอร์-command-line)
2. [พารามิเตอร์ RSL-RL Agent Configuration](#พารามิเตอร์-rsl-rl-agent-configuration)
3. [พารามิเตอร์ Environment Configuration](#พารามิเตอร์-environment-configuration)
4. [พารามิเตอร์ Reward Functions](#พารามิเตอร์-reward-functions)

---

## พารามิเตอร์ Command Line

### 1. การจัดการ Environment

#### `--task TASK_NAME`
- **ความหมาย**: ระบุชื่อ task/environment ที่ต้องการเทรน
- **ค่าตัวอย่าง**: `Isaac-Velocity-Flat-Humanoid-10102025-v0`
- **ประเภท**: String
- **จำเป็น**: ✅ ใช่
- **หน้าที่**:
  - กำหนด environment ที่จะใช้ในการเทรน
  - ต้องตรงกับชื่อที่ register ไว้ใน `gym.register()`
  - แต่ละ task มี configuration แยกกันเช่น rewards, observations, terrain

#### `--num_envs NUM_ENVS`
- **ความหมาย**: จำนวน parallel environments ที่จะรันพร้อมกัน
- **ค่าตัวอย่าง**: `4096`, `2048`, `1024`
- **ค่า default**: ขึ้นอยู่กับ environment config
- **ประเภท**: Integer
- **หน้าที่**:
  - **การเก็บข้อมูล**: เก็บข้อมูล experience มากขึ้นต่อ iteration
  - **ความเร็ว**: เพิ่มจำนวนทำให้เทรนเร็วขึ้น (ถ้า GPU รองรับ)
  - **Stability**: จำนวนมากช่วยให้การเรียนรู้เสถียรขึ้น
  - **Trade-off**: ใช้ GPU memory มากขึ้น
- **แนะนำ**:
  - GPU 8 GB → `512-1024` envs
  - GPU 12 GB → `2048` envs
  - GPU 16 GB → `4096` envs
  - GPU 24 GB+ → `8192` envs

#### `--seed SEED`
- **ความหมาย**: Random seed สำหรับการสุ่มตัวเลข
- **ค่าตัวอย่าง**: `42`, `123`, `2024`
- **ค่า default**: `None` (random)
- **ประเภท**: Integer
- **หน้าที่**:
  - **Reproducibility**: ทำให้สามารถทำซ้ำผลการทดลองได้
  - **Initialization**: กำหนด random initialization ของ neural network
  - **Sampling**: ควบคุมการสุ่มของ environment randomization
- **แนะนำ**: ใช้ค่าเดียวกันเมื่อต้องการเปรียบเทียบผล

### 2. การจัดการการเทรน

#### `--max_iterations MAX_ITERATIONS`
- **ความหมาย**: จำนวนรอบการเทรนทั้งหมด
- **ค่าตัวอย่าง**: `3000`, `2000`, `1000`
- **ค่า default**: `3000` (จาก agent config)
- **ประเภท**: Integer
- **หน้าที่**:
  - **Training Duration**: กำหนดความยาวของการเทรน
  - **Convergence**: ค่าสูงให้เวลา policy เรียนรู้มากขึ้น
  - **การคำนวณ**: Total steps = `max_iterations × num_steps_per_env × num_envs`
- **ตัวอย่าง**:
  - 3000 iterations × 24 steps × 4096 envs = 295,321,600 steps
- **แนะนำ**:
  - Flat terrain → `1000-2000` iterations
  - Rough terrain → `3000-5000` iterations

#### `--resume`
- **ความหมาย**: ต่อการเทรนจาก checkpoint ที่มีอยู่
- **ค่าตัวอย่าง**: flag (ไม่มีค่า)
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - **Continue Training**: เทรนต่อจากที่หยุดไว้
  - **Recovery**: กู้คืนการเทรนที่ขาดหาย
  - **Fine-tuning**: ปรับแต่ง policy ที่เทรนแล้ว
- **ใช้ร่วมกับ**: `--load_run` หรือ `--checkpoint`

#### `--load_run LOAD_RUN`
- **ความหมาย**: path ไปยังโฟลเดอร์ของ run ที่ต้องการโหลด
- **ค่าตัวอย่าง**: `logs/rsl_rl/humanoid_10102025/2025-01-24_10-30-45`
- **ประเภท**: String (path)
- **หน้าที่**:
  - **Load Checkpoint**: โหลด model weights และ optimizer state
  - **Continue**: ต่อเทรนจากรอบที่หยุดไว้
  - **Transfer Learning**: ใช้ pre-trained weights
- **โครงสร้างโฟลเดอร์**:
  ```
  logs/rsl_rl/humanoid_10102025/YYYY-MM-DD_HH-MM-SS/
  ├── model_<iteration>.pt
  ├── config.yaml
  └── summaries/
  ```

#### `--checkpoint CHECKPOINT`
- **ความหมาย**: path ไปยังไฟล์ checkpoint เฉพาะ
- **ค่าตัวอย่าง**: `model_2000.pt`
- **ประเภท**: String (filename)
- **หน้าที่**:
  - **Specific Checkpoint**: โหลดจาก iteration ที่ระบุ
  - **Best Model**: เลือก checkpoint ที่ได้ผลดีที่สุด
- **ใช้ร่วมกับ**: `--load_run`

#### `--distributed`
- **ความหมาย**: เปิดใช้การเทรนแบบ distributed
- **ค่าตัวอย่าง**: flag (ไม่มีค่า)
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - **Multi-GPU**: กระจายการคำนวณไปยังหลาย GPUs
  - **Scalability**: เทรนด้วย environments จำนวนมาก
  - **Speed**: เพิ่มความเร็วการเทรน
- **ข้อกำหนด**: ต้องมี multiple GPUs
- **Implementation**: ใช้ PyTorch Distributed Data Parallel (DDP)

### 3. Logging และ Monitoring

#### `--logger {tensorboard,wandb,neptune}`
- **ความหมาย**: เลือก logging framework
- **ค่าตัวอย่าง**:
  - `tensorboard` (default)
  - `wandb` (Weights & Biases)
  - `neptune` (Neptune.ai)
- **ประเภท**: Choice
- **หน้าที่**:
  - **TensorBoard**:
    - Local logging
    - Real-time monitoring
    - Built-in visualization
    - ไม่ต้องลงทะเบียน
  - **Weights & Biases**:
    - Cloud-based
    - Team collaboration
    - Experiment comparison
    - Hyperparameter sweeps
  - **Neptune.ai**:
    - Enterprise features
    - Advanced experiment management
    - Model versioning
- **การใช้งาน**:
  ```bash
  # TensorBoard (local)
  tensorboard --logdir logs/rsl_rl/humanoid_10102025/

  # W&B (requires login)
  wandb login
  ```

#### `--log_project_name LOG_PROJECT_NAME`
- **ความหมาย**: ชื่อโปรเจกต์สำหรับ wandb/neptune
- **ค่าตัวอย่าง**: `humanoid_locomotion`, `bipedal_walking`
- **ประเภท**: String
- **หน้าที่**:
  - **Organization**: จัดกลุ่ม experiments
  - **Team Work**: แชร์ผลกับทีม
  - **Filtering**: กรองและค้นหา runs
- **ใช้กับ**: `--logger wandb` หรือ `--logger neptune`

#### `--experiment_name EXPERIMENT_NAME`
- **ความหมาย**: ชื่อ experiment folder
- **ค่าตัวอย่าง**: `humanoid_10102025`, `test_run`
- **ค่า default**: ชื่อ task
- **ประเภท**: String
- **หน้าที่**:
  - **Directory**: สร้างโฟลเดอร์ใน `logs/rsl_rl/`
  - **Grouping**: จัดกลุ่ม runs ที่เกี่ยวข้อง
- **ผลลัพธ์**: `logs/rsl_rl/{experiment_name}/YYYY-MM-DD_HH-MM-SS/`

#### `--run_name RUN_NAME`
- **ความหมาย**: ชื่อเพิ่มเติมสำหรับแต่ละ run
- **ค่าตัวอย่าง**: `trial_01`, `high_lr`, `test`
- **ประเภท**: String
- **หน้าที่**:
  - **Identification**: ระบุลักษณะเฉพาะของ run
  - **Suffix**: เพิ่มท้าย timestamp
- **ผลลัพธ์**: `logs/rsl_rl/experiment/YYYY-MM-DD_HH-MM-SS_{run_name}/`

### 4. การบันทึกวิดีโอ

#### `--video`
- **ความหมาย**: เปิดการบันทึกวิดีโอระหว่างเทรน
- **ค่าตัวอย่าง**: flag (ไม่มีค่า)
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - **Visualization**: ดูพฤติกรรม robot ระหว่างเทรน
  - **Progress Monitoring**: ติดตามความก้าวหน้า
  - **Documentation**: บันทึกผลการทดลอง
- **Trade-off**: ใช้ storage space และชะลอการเทรนเล็กน้อย

#### `--video_length VIDEO_LENGTH`
- **ความหมาย**: ความยาววิดีโอ (จำนวน steps)
- **ค่าตัวอย่าง**: `200`, `300`, `500`
- **ค่า default**: `200`
- **ประเภท**: Integer
- **หน้าที่**:
  - **Duration**: กำหนดความยาวของแต่ละวิดีโอ
  - **File Size**: ค่าสูงทำให้ไฟล์ใหญ่ขึ้น
- **การคำนวณ**: เวลาจริง (วินาที) = `video_length × dt × decimation`
  - ตัวอย่าง: 200 steps × 0.005 s × 2 = 2 วินาที

#### `--video_interval VIDEO_INTERVAL`
- **ความหมาย**: ช่วงห่างระหว่างการบันทึกวิดีโอ (steps)
- **ค่าตัวอย่าง**: `1000`, `500`, `2000`
- **ค่า default**: `1000`
- **ประเภท**: Integer
- **หน้าที่**:
  - **Frequency**: กำหนดความถี่ในการบันทึก
  - **Storage**: ค่าสูงประหยัด storage
  - **Monitoring**: ค่าต่ำเห็นความก้าวหน้าชัดเจนขึ้น

### 5. Simulation และ Rendering

#### `--headless`
- **ความหมาย**: ปิด GUI/visualization
- **ค่าตัวอย่าง**: flag (ไม่มีค่า)
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - **Performance**: เพิ่มความเร็วการเทรน 2-3 เท่า
  - **Server Mode**: เหมาะสำหรับ remote servers
  - **Resource Saving**: ลด GPU usage สำหรับ rendering
- **แนะนำ**: ใช้เสมอเมื่อเทรนจริง ๆ

#### `--enable_cameras`
- **ความหมาย**: เปิดใช้งาน camera sensors
- **ค่าตัวอย่าง**: flag (ไม่มีค่า)
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - **Vision-based RL**: เทรนด้วย visual observations
  - **RGB/Depth**: เปิดใช้ camera data
- **Trade-off**: ใช้ resources มากขึ้น

#### `--device DEVICE`
- **ความหมาย**: เลือก device สำหรับการคำนวณ
- **ค่าตัวอย่าง**:
  - `cuda:0` (GPU แรก)
  - `cuda:1` (GPU ที่สอง)
  - `cpu` (CPU mode)
- **ค่า default**: `cuda:0`
- **ประเภท**: String
- **หน้าที่**:
  - **GPU Selection**: เลือก GPU เฉพาะ
  - **Multi-GPU**: กระจาย workload
- **การตรวจสอบ**:
  ```bash
  nvidia-smi  # ดู GPU utilization
  ```

#### `--livestream {0,1,2}`
- **ความหมาย**: เปิด livestreaming
- **ค่าตัวอย่าง**: `0` (off), `1` (native), `2` (websocket)
- **ประเภท**: Integer (choice)
- **หน้าที่**:
  - **Remote Viewing**: ดูการเทรนจากเครื่องอื่น
  - **Web Browser**: เข้าถึงผ่าน browser
- **Port**: ดูที่ `localhost:8211/streaming/webrtc-client/`

#### `--rendering_mode {performance,balanced,quality,xr}`
- **ความหมาย**: โหมด rendering quality
- **ค่าตัวอย่าง**:
  - `performance` - เร็วที่สุด, คุณภาพต่ำสุด
  - `balanced` - สมดุล (default)
  - `quality` - คุณภาพสูง, ช้าที่สุด
  - `xr` - สำหรับ VR/AR
- **ประเภท**: Choice
- **หน้าที่**:
  - **FPS**: กำหนด frames per second
  - **Visual Quality**: ความสวยงามของภาพ
- **แนะนำ**: ใช้ `performance` เมื่อเทรนโดยไม่ใช้ `--headless`

#### `--verbose` / `--info`
- **ความหมาย**: ระดับของ logging output
- **ประเภท**: Boolean flag
- **หน้าที่**:
  - `--verbose`: แสดง debug messages ทั้งหมด
  - `--info`: แสดง info level messages
- **ใช้เมื่อ**: Debug หรือต้องการดูรายละเอียด

---

## พารามิเตอร์ RSL-RL Agent Configuration

Configuration เหล่านี้อยู่ในไฟล์ `agents/rsl_rl_ppo_cfg.py`

### 1. Runner Configuration

#### `num_steps_per_env`
- **ค่า**: `24`
- **ความหมาย**: จำนวน steps ที่เก็บต่อ environment ต่อ iteration
- **หน้าที่**:
  - **Experience Collection**: กำหนดขนาดของ rollout buffer
  - **Update Frequency**: บ่อยแค่ไหนที่จะ update policy
  - **Total Steps per Iteration**: `num_steps_per_env × num_envs`
- **ตัวอย่าง**: 24 steps × 4096 envs = 98,304 steps per iteration
- **Trade-off**:
  - ค่าสูง → เก็บข้อมูลมากขึ้น, update ช้าลง
  - ค่าต่ำ → update บ่อยขึ้น, อาจไม่เสถียร

#### `max_iterations`
- **ค่า**: `3000`
- **ความหมาย**: จำนวน iterations สูงสุดที่จะเทรน
- **หน้าที่**:
  - **Training Duration**: กำหนดความยาวการเทรน
  - **Total Training Steps**: `max_iterations × num_steps_per_env × num_envs`
- **ตัวอย่าง**: 3000 × 24 × 4096 = 295,321,600 total steps
- **เวลาเทรน**: ~3-4 ชั่วโมงบน RTX 4090 (4096 envs, headless)

#### `save_interval`
- **ค่า**: `50`
- **ความหมาย**: บันทึก checkpoint ทุก ๆ กี่ iteration
- **หน้าที่**:
  - **Checkpoint Frequency**: กำหนดความถี่การบันทึก model
  - **Recovery**: สามารถกู้คืนจากจุดต่าง ๆ
  - **Storage**: ค่าต่ำใช้ storage มากขึ้น
- **ตัวอย่าง**: 50 iterations → บันทึก 60 checkpoints (3000/50)

#### `experiment_name`
- **ค่า**: `"humanoid_10102025"`
- **ความหมาย**: ชื่อ experiment
- **หน้าที่**:
  - **Directory Naming**: สร้างโฟลเดอร์ logs
  - **Organization**: จัดกลุ่ม runs
- **ผลลัพธ์**: `logs/rsl_rl/humanoid_10102025/`

#### `empirical_normalization`
- **ค่า**: `False`
- **ความหมาย**: ใช้ empirical normalization หรือไม่
- **หน้าที่**:
  - **Observation Normalization**: ปรับ scale ของ observations
  - `True`: ใช้ running mean/std จาก experience
  - `False`: ใช้ normalization ที่กำหนดไว้ใน environment
- **แนะนำ**: `False` สำหรับ humanoid (ใช้ normalization ที่ tune แล้ว)

### 2. Policy Network Configuration

#### `init_noise_std`
- **ค่า**: `1.0`
- **ความหมาย**: Standard deviation ของ noise เริ่มต้น
- **หน้าที่**:
  - **Exploration**: เพิ่ม exploration ในช่วงแรก
  - **Action Noise**: เพิ่ม randomness ให้ actions
  - **Decay**: ค่านี้จะลดลงตาม training schedule
- **ผลกระทบ**:
  - ค่าสูง → exploration มากขึ้น, ไม่เสถียรในช่วงแรก
  - ค่าต่ำ → exploitation เร็วขึ้น, อาจติด local optima

#### `actor_hidden_dims`
- **ค่า**: `[512, 256, 128]`
- **ความหมาย**: ขนาดของ hidden layers ใน actor network
- **หน้าที่**:
  - **Network Capacity**: กำหนดความสามารถในการเรียนรู้
  - **Layer Structure**:
    - Input layer → 512 neurons
    - Hidden layer 1 → 256 neurons
    - Hidden layer 2 → 128 neurons
    - Output layer → num_actions (21 สำหรับ humanoid)
- **Total Parameters**: ~การคำนวณขึ้นอยู่กับ input size
  - สมมติ input = 75: (75×512) + (512×256) + (256×128) + (128×21) ≈ 204,000 parameters
- **Trade-off**:
  - ใหญ่ขึ้น → เรียนรู้ได้ซับซ้อนขึ้น, ใช้ memory มากขึ้น, อาจ overfit
  - เล็กลง → เร็วขึ้น, ใช้ memory น้อยลง, อาจเรียนรู้ไม่เพียงพอ

#### `critic_hidden_dims`
- **ค่า**: `[512, 256, 128]`
- **ความหมาย**: ขนาดของ hidden layers ใน critic network
- **หน้าที่**:
  - **Value Estimation**: ประมาณค่า value function V(s)
  - **Advantage Calculation**: ใช้คำนวณ advantage = Q(s,a) - V(s)
- **โครงสร้าง**: เหมือน actor แต่ output = 1 (value)
- **หมายเหตุ**: มักตั้งเท่ากับ actor เพื่อความสมดุล

#### `activation`
- **ค่า**: `"elu"`
- **ความหมาย**: Activation function สำหรับ hidden layers
- **ตัวเลือก**:
  - `"elu"` (Exponential Linear Unit) - smooth, ไม่มี dying ReLU problem
  - `"relu"` - เร็วที่สุด, standard
  - `"selu"` - self-normalizing
  - `"tanh"` - bounded output
- **ELU Formula**: `f(x) = x if x > 0 else α(e^x - 1)`
- **ข้อดี ELU**:
  - Smooth gradient
  - Negative values (mean activation ~0)
  - ป้องกัน dying neurons
- **ใช้กับ Humanoid**: ELU ดีสำหรับ continuous control

### 3. PPO Algorithm Configuration

#### `value_loss_coef`
- **ค่า**: `1.0`
- **ความหมาย**: น้ำหนักของ value loss ใน total loss
- **สูตร**: `total_loss = policy_loss + value_loss_coef × value_loss - entropy_coef × entropy`
- **หน้าที่**:
  - **Value Learning**: กำหนดความสำคัญของการเรียนรู้ value function
  - **Balance**: ปรับสมดุลระหว่าง policy และ value learning
- **ผลกระทบ**:
  - ค่าสูง → critic เรียนรู้เร็วขึ้น, advantage estimation แม่นยำขึ้น
  - ค่าต่ำ → โฟกัสที่ policy มากขึ้น

#### `use_clipped_value_loss`
- **ค่า**: `True`
- **ความหมาย**: ใช้ clipped value loss หรือไม่
- **หน้าที่**:
  - **Stability**: ป้องกัน value function เปลี่ยนแปลงรุนแรง
  - **PPO-style**: เหมือน policy clipping แต่สำหรับ value
- **สูตร**:
  ```
  clipped_value = old_value + clip(new_value - old_value, -ε, +ε)
  value_loss = max(MSE(new_value, target), MSE(clipped_value, target))
  ```
- **แนะนำ**: ใช้ `True` สำหรับความเสถียร

#### `clip_param`
- **ค่า**: `0.2`
- **ความหมาย**: ε สำหรับ PPO clipping
- **หน้าที่**:
  - **Trust Region**: จำกัดการเปลี่ยนแปลงของ policy
  - **Stability**: ป้องกัน destructive updates
- **สูตร**:
  ```
  ratio = π_new(a|s) / π_old(a|s)
  clipped_ratio = clip(ratio, 1-ε, 1+ε)
  policy_loss = -min(ratio × advantage, clipped_ratio × advantage)
  ```
- **ค่า 0.2 หมายความว่า**:
  - Policy สามารถเปลี่ยนแปลงได้ไม่เกิน ±20%
  - ช่วง ratio: [0.8, 1.2]
- **ผลกระทบ**:
  - ค่าสูง (0.3) → update ใหญ่ขึ้น, เรียนเร็วแต่อาจไม่เสถียร
  - ค่าต่ำ (0.1) → update เล็กลง, ช้าแต่เสถียรกว่า

#### `entropy_coef`
- **ค่า**: `0.01`
- **ความหมาย**: น้ำหนักของ entropy bonus
- **หน้าที่**:
  - **Exploration**: ส่งเสริมให้ policy มีความหลากหลาย
  - **Prevent Premature Convergence**: ป้องกัน policy เป็น deterministic เร็วเกินไป
- **สูตร**: `entropy = -Σ π(a|s) log π(a|s)`
- **สูตร total loss**: `loss = policy_loss + value_loss - 0.01 × entropy`
- **ผลกระทบ**:
  - ค่าสูง (0.05) → exploration มากขึ้น, policy random มากขึ้น
  - ค่าต่ำ (0.001) → exploitation มากขึ้น, policy deterministic เร็วขึ้น
- **Decay**: ค่านี้อาจถูกลดตาม schedule

#### `num_learning_epochs`
- **ค่า**: `5`
- **ความหมาย**: จำนวน epochs ที่จะ optimize ต่อ iteration
- **หน้าที่**:
  - **Sample Efficiency**: ใช้ซ้ำ experience data
  - **Gradient Updates**: จำนวนครั้งที่ update จาก experience buffer
- **การทำงาน**:
  - เก็บ 24×4096 = 98,304 transitions
  - แบ่งเป็น 4 mini-batches
  - Loop 5 epochs → 5×4 = 20 gradient updates per iteration
- **ผลกระทบ**:
  - ค่าสูง (10) → sample efficient มากขึ้น, แต่อาจ overfit data
  - ค่าต่ำ (1) → underutilize data, ต้อง collect data มากขึ้น

#### `num_mini_batches`
- **ค่า**: `4`
- **ความหมาย**: แบ่ง experience buffer เป็นกี่ mini-batches
- **หน้าที่**:
  - **Batch Size**: mini_batch_size = (num_steps_per_env × num_envs) / num_mini_batches
  - **Memory Management**: แบ่งเพื่อให้พอดี GPU memory
  - **Gradient Variance**: mini-batch เล็กทำให้ gradient มี variance สูงขึ้น
- **การคำนวณ**:
  - Total transitions = 24 × 4096 = 98,304
  - Mini-batch size = 98,304 / 4 = 24,576 transitions
- **ผลกระทบ**:
  - ค่าสูง (8) → mini-batch เล็กลง, update บ่อยขึ้น, gradient noisy ขึ้น
  - ค่าต่ำ (2) → mini-batch ใหญ่ขึ้น, gradient smooth ขึ้น

#### `learning_rate`
- **ค่า**: `1.0e-3` (0.001)
- **ความหมาย**: ขนาดของ gradient step
- **หน้าที่**:
  - **Update Size**: กำหนดขนาดการเปลี่ยนแปลง network weights
  - **Convergence Speed**: มีผลต่อความเร็วในการเรียนรู้
- **สูตร**: `θ_new = θ_old - learning_rate × gradient`
- **ผลกระทบ**:
  - ค่าสูง (1e-2) → เรียนเร็วขึ้น, แต่อาจไม่ converge หรือ unstable
  - ค่าต่ำ (1e-4) → เรียนช้าลง, แต่ stable และ precise ขึ้น
- **Adaptive**: ใช้ร่วมกับ `schedule="adaptive"` เพื่อปรับตาม KL divergence

#### `schedule`
- **ค่า**: `"adaptive"`
- **ความหมาย**: วิธีการปรับ learning rate
- **ตัวเลือก**:
  - `"adaptive"` - ปรับตาม KL divergence
  - `"linear"` - ลดเชิงเส้นจาก initial → 0
  - `"constant"` - คงที่ตลอด
- **Adaptive Schedule**:
  - วัด KL divergence ระหว่าง old และ new policy
  - ถ้า KL > desired_kl × 2 → lr = lr / 1.5 (ลด)
  - ถ้า KL < desired_kl / 2 → lr = lr × 1.5 (เพิ่ม)
  - จำกัด lr ในช่วง [1e-5, 1e-2]
- **ข้อดี**: ปรับตัวตามความเหมาะสม, ไม่ต้อง tune manual

#### `gamma`
- **ค่า**: `0.99`
- **ความหมาย**: Discount factor สำหรับ future rewards
- **หน้าที่**:
  - **Time Horizon**: กำหนดว่าให้ความสำคัญกับ future มากน้อยแค่ไหน
  - **Return Calculation**: `G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...`
- **ค่า 0.99 หมายความว่า**:
  - Reward หลัง 1 step มีค่า 99% ของ immediate reward
  - Reward หลัง 10 steps มีค่า ~90% (0.99^10)
  - Reward หลัง 100 steps มีค่า ~37% (0.99^100)
- **Effective Horizon**: `1/(1-γ) = 1/0.01 = 100 steps`
- **ผลกระทบ**:
  - ค่าสูง (0.999) → horizon ยาวขึ้น (1000 steps), long-term planning
  - ค่าต่ำ (0.95) → horizon สั้นลง (20 steps), short-term rewards
- **สำหรับ Humanoid**: 0.99 เหมาะสมเพราะต้องการ long-term stability

#### `lam` (Lambda for GAE)
- **ค่า**: `0.95`
- **ความหมาย**: Lambda สำหรับ Generalized Advantage Estimation (GAE)
- **หน้าที่**:
  - **Bias-Variance Trade-off**: ปรับสมดุลระหว่าง bias และ variance ของ advantage
  - **Multi-step Returns**: ผสม TD-error หลาย steps
- **สูตร GAE**:
  ```
  δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD-error)
  A_t = Σ (γλ)^k δ_{t+k}            (GAE advantage)
  ```
- **ความหมายของ λ**:
  - λ = 0 → ใช้ 1-step TD (bias สูง, variance ต่ำ)
  - λ = 1 → ใช้ Monte Carlo (bias ต่ำ, variance สูง)
  - λ = 0.95 → ผสมทั้งสอง (balanced)
- **Effective Horizon**: `1/(1-γλ) ≈ 20 steps`
- **ผลกระทบ**:
  - ค่าสูง (0.99) → smooth advantage, variance ต่ำลง
  - ค่าต่ำ (0.90) → responsive advantage, อาจมี noise

#### `desired_kl`
- **ค่า**: `0.01`
- **ความหมาย**: Target KL divergence ระหว่าง old และ new policy
- **หน้าที่**:
  - **Trust Region**: จำกัดการเปลี่ยนแปลงของ policy
  - **Adaptive Learning Rate**: ใช้ปรับ learning rate อัตโนมัติ
- **สูตร KL**: `KL(π_old || π_new) = Σ π_old(a|s) log(π_old(a|s) / π_new(a|s))`
- **ความหมาย 0.01**:
  - Policy เปลี่ยนแปลงเล็กน้อย per update
  - Safe, conservative updates
- **การใช้งาน**:
  - วัด KL หลัง gradient update
  - ถ้า KL >> 0.01 → ลด learning rate
  - ถ้า KL << 0.01 → เพิ่ม learning rate
- **Early Stopping**: ถ้า KL > 0.01 × 1.5 → หยุด epoch ก่อนกำหนด

#### `max_grad_norm`
- **ค่า**: `1.0`
- **ความหมาย**: Maximum gradient norm สำหรับ gradient clipping
- **หน้าที่**:
  - **Prevent Exploding Gradients**: จำกัดขนาดของ gradient
  - **Training Stability**: ป้องกัน catastrophic updates
- **การทำงาน**:
  ```python
  grad_norm = ||∇θ||₂  # L2 norm of gradients
  if grad_norm > max_grad_norm:
      ∇θ = ∇θ × (max_grad_norm / grad_norm)
  ```
- **ตัวอย่าง**:
  - ถ้า gradient norm = 5.0
  - Clip to: gradient × (1.0 / 5.0) = gradient × 0.2
- **ผลกระทบ**:
  - ค่าสูง (10.0) → ยอมให้ gradient ใหญ่ขึ้น, อาจ unstable
  - ค่าต่ำ (0.5) → จำกัด gradient แน่นขึ้น, เรียนช้าลงแต่ stable

---

## พารามิเตอร์ Environment Configuration

Configuration ใน `rough_env_cfg.py`

### 1. Episode Settings

#### `episode_length_s`
- **ค่า**: ขึ้นอยู่กับ base config (~20 วินาที)
- **ความหมาย**: ความยาวของแต่ละ episode (วินาที)
- **หน้าที่**:
  - **Task Duration**: กำหนดความยาวงานก่อน reset
  - **Horizon**: มีผลต่อการเรียนรู้ long-term behavior
- **การคำนวณ steps**: `episode_length_s / (dt × decimation)`
  - ตัวอย่าง: 20s / (0.005s × 2) = 2000 steps per episode

#### `decimation`
- **ค่า**: ขึ้นอยู่กับ base config (~2)
- **ความหมาย**: จำนวน physics steps ต่อ control step
- **หน้าที่**:
  - **Control Frequency**: ลด control frequency
  - **Stability**: เพิ่ม stability ของการควบคุม
- **ตัวอย่าง**:
  - Physics: 200 Hz (dt = 0.005s)
  - Decimation = 2
  - Control: 100 Hz (0.01s)

### 2. Observation และ Action Space

#### `num_actions`
- **ค่า**: จำนวน DOF ของ robot
- **สำหรับ Humanoid-10102025**: 21 DOF
  - 6 DOF × 2 legs = 12
  - 4 DOF × 2 arms = 8
  - 1 DOF pelvis = 1
- **หน้าที่**: กำหนดขนาด output ของ actor network

#### `num_observations`
- **ค่า**: ขนาดของ observation vector
- **สำหรับ Humanoid**: ~75 dimensions
  - Base orientation (4 - quaternion)
  - Base linear velocity (3)
  - Base angular velocity (3)
  - Joint positions (21)
  - Joint velocities (21)
  - Previous actions (21)
  - Commanded velocities (3)
- **หน้าที่**: กำหนดขนาด input ของ networks

### 3. Scene Configuration

#### `num_envs`
- **ค่า**: จำนวน parallel environments
- **กำหนดโดย**: command line `--num_envs`
- **Trade-off**: ดูที่ [--num_envs](#--num_envs-num_envs)

#### `env_spacing`
- **ค่า**: ~4.0 เมตร
- **ความหมาย**: ระยะห่างระหว่าง environments
- **หน้าที่**:
  - **Collision Prevention**: ป้องกัน robots ชน environments อื่น
  - **Visualization**: แยก environments ให้เห็นชัด

### 4. Terrain Configuration

#### `terrain_type`
- **ตัวเลือก**:
  - `"flat"` - พื้นราบ
  - `"rough"` - พื้นขรุขระ (มีอุปสรรค)
- **สำหรับ Humanoid-10102025-v0**: "flat"
- **หน้าที่**:
  - **Task Difficulty**: กำหนดความยาก
  - **Curriculum**: เริ่มจาก flat → rough

### 5. Command Ranges

#### `lin_vel_x` (Linear Velocity X)
- **ค่า**: `(0.0, 1.0)` m/s
- **ความหมาย**: ช่วงความเร็วเดินไปข้างหน้า
- **หน้าที่**:
  - **Velocity Command**: สุ่มคำสั่งความเร็วในช่วงนี้
  - **Training Diversity**: เทรนหลายความเร็ว
- **หมายเหตุ**: 0-1 m/s เหมาะสำหรับการเดิน

#### `lin_vel_y` (Linear Velocity Y)
- **ค่า**: `(-0.0, 0.0)` m/s (disabled)
- **ความหมาย**: ความเร็วด้านข้าง
- **หน้าที่**: ปิดการเดินด้านข้างในช่วงแรก

#### `ang_vel_z` (Angular Velocity Z)
- **ค่า**: `(-1.0, 1.0)` rad/s
- **ความหมาย**: ความเร็วการหมุน (yaw)
- **หน้าที่**:
  - **Turning**: เทรนการหมุนตัว
  - **Range**: ~±57 องศา/วินาที

---

## พารามิเตอร์ Reward Functions

Configuration ใน `Humanoid10102025Rewards`

### 1. Tracking Rewards

#### `track_lin_vel_xy_exp`
- **น้ำหนัก**: `1.0`
- **ฟังก์ชัน**: `mdp.track_lin_vel_xy_yaw_frame_exp`
- **พารามิเตอร์**: `std=0.5`
- **หน้าที่**:
  - **Velocity Tracking**: รางวัลสำหรับติดตามความเร็วที่สั่ง
  - **Exponential**: ใช้ exp(-error²/(2σ²))
- **สูตร**:
  ```
  error = ||v_commanded - v_actual||₂
  reward = exp(-error² / (2 × 0.5²))
  ```
- **ค่า std=0.5**:
  - Tolerance: ±0.5 m/s ถือว่าใกล้เคียง
  - Steep penalty สำหรับ error > 0.5

#### `track_ang_vel_z_exp`
- **น้ำหนัก**: `2.0`
- **ฟังก์ชัน**: `mdp.track_ang_vel_z_world_exp`
- **หน้าที่**: ติดตามความเร็วการหมุน
- **น้ำหนัก 2.0**: การหมุนสำคัญกว่าการเดิน 2 เท่า

### 2. Gait Rewards

#### `feet_air_time`
- **น้ำหนัก**: `0.25`
- **ฟังก์ชัน**: `mdp.feet_air_time_positive_biped`
- **พารามิเตอร์**: `threshold=0.4`
- **หน้าที่**:
  - **Encourage Walking**: ให้รางวัลเมื่อเท้าลอยตัว
  - **Gait Pattern**: ส่งเสริม natural gait
- **threshold 0.4 วินาที**: เท้าต้องลอยอย่างน้อย 0.4s

#### `feet_slide`
- **น้ำหนัก**: `-0.1` (penalty)
- **ฟังก์ชัน**: `mdp.feet_slide`
- **หน้าที่**:
  - **Prevent Sliding**: ลงโทษการเลื่อนเท้า
  - **Realistic Contact**: ส่งเสริม proper foot placement
- **Penalty**: ยิ่งเลื่อนมาก ยิ่งโดน penalty มาก

### 3. Safety Rewards

#### `termination_penalty`
- **น้ำหนัก**: `-200.0` (penalty)
- **ฟังก์ชัน**: `mdp.is_terminated`
- **หน้าที่**:
  - **Avoid Termination**: penalty มากเมื่อล้ม
  - **Safety**: ส่งเสริมความปลอดภัย
- **ค่า -200**: penalty ใหญ่มาก ให้หลีกเลี่ยงการล้ม

#### `flat_orientation_l2`
- **น้ำหนัก**: `-1.0` (penalty)
- **หน้าที่**:
  - **Upright Posture**: ลงโทษการเอียง
  - **L2 norm**: penalty ตามมุมเอียง²
- **Target**: ให้ robot ยืนตรง

### 4. Regularization Rewards

#### `action_rate_l2`
- **น้ำหนัก**: `-0.005` (penalty)
- **หน้าที่**:
  - **Smooth Actions**: ลงโทษ action เปลี่ยนรวดเร็ว
  - **L2**: penalty = ||a_t - a_{t-1}||²
- **ผล**: actions นุ่มนวลขึ้น, ไม่กระตุก

#### `dof_acc_l2`
- **น้ำหนัก**: `-1.25e-7` (penalty เล็กมาก)
- **หน้าที่**:
  - **Smooth Motion**: ลงโทษความเร่งของ joints
  - **Apply to**: hip และ knee joints
- **ค่าเล็กมาก**: regularization อ่อน ๆ

#### `dof_torques_l2`
- **น้ำหนัก**: `-1.5e-7` (penalty เล็กมาก)
- **หน้าที่**:
  - **Energy Efficiency**: ลงโทษ torque สูง
  - **Apply to**: hip, knee, ankle
- **ผล**: ประหยัดพลังงาน

### 5. Joint Limit Rewards

#### `dof_pos_limits`
- **น้ำหนัก**: `-1.0` (penalty)
- **หน้าที่**:
  - **Avoid Limits**: ลงโทษเมื่อใกล้ joint limits
  - **Safety**: ป้องกันความเสียหาย
- **Apply to**: ankle joints (อ่อนไหวที่สุด)

#### `joint_deviation_hip`
- **น้ำหนัก**: `-0.1` (penalty)
- **หน้าที่**:
  - **Natural Pose**: ลงโทษการเบี่ยงเบนจาก default pose
  - **Apply to**: hip_yaw และ hip_roll
- **L1 norm**: penalty เชิงเส้น

#### `joint_deviation_arms`
- **น้ำหนัก**: `-0.1` (penalty)
- **หน้าที่**:
  - **Arm Position**: ให้แขนอยู่ในท่าธรรมชาติ
  - **Apply to**: shoulder และ elbow joints
- **ผล**: แขนไม่โบกวูบมากเกินไป

---

## สรุปความสัมพันธ์ระหว่างพารามิเตอร์

### Training Speed vs Quality

| พารามิเตอร์ | เพิ่มความเร็ว | เพิ่มคุณภาพ |
|------------|---------------|-------------|
| `--num_envs` | ↑ (4096-8192) | ↑ (more data) |
| `--headless` | ✅ (~2-3x) | - |
| `num_learning_epochs` | ↓ (1-3) | ↑ (5-10) |
| `num_mini_batches` | ↓ (2) | ↑ (8) |
| `actor_hidden_dims` | ↓ [256,128] | ↑ [512,512,256] |

### Exploration vs Exploitation

| พารามิเตอร์ | เพิ่ม Exploration | เพิ่ม Exploitation |
|------------|-------------------|-------------------|
| `init_noise_std` | ↑ (1.5) | ↓ (0.5) |
| `entropy_coef` | ↑ (0.05) | ↓ (0.001) |
| `clip_param` | ↑ (0.3) | ↓ (0.1) |

### Stability vs Speed

| พารามิเตอร์ | เพิ่ม Stability | เพิ่ม Speed |
|------------|-----------------|------------|
| `learning_rate` | ↓ (1e-4) | ↑ (5e-3) |
| `clip_param` | ↓ (0.1) | ↑ (0.3) |
| `desired_kl` | ↓ (0.005) | ↑ (0.02) |
| `max_grad_norm` | ↓ (0.5) | ↑ (5.0) |

---

## แนะนำการปรับแต่ง (Tuning Guidelines)

### สำหรับ Humanoid Walking

**Conservative (Stable)**:
```bash
--num_envs 2048 --max_iterations 5000
```
Config changes:
- `learning_rate = 5e-4`
- `clip_param = 0.15`
- `entropy_coef = 0.008`

**Aggressive (Fast)**:
```bash
--num_envs 8192 --max_iterations 2000
```
Config changes:
- `learning_rate = 2e-3`
- `clip_param = 0.25`
- `num_learning_epochs = 8`

**Balanced (Recommended)**:
```bash
--num_envs 4096 --max_iterations 3000
```
- ใช้ค่า default ทั้งหมด

---

## การคำนวณ Training Time

### สูตร:
```
Total Steps = max_iterations × num_steps_per_env × num_envs
Wall Time = Total Steps / (FPS × num_envs)
```

### ตัวอย่าง (RTX 4090, headless):
- FPS ≈ 60,000-80,000 (with 4096 envs)
- Total Steps = 3000 × 24 × 4096 = 295,321,600
- Training Time = 295M / (70k × 4096) ≈ 3-4 ชั่วโมง

### ปัจจัยที่มีผล:
1. **GPU**: RTX 4090 > RTX 3090 > RTX 3080
2. **Headless**: ~2-3x เร็วกว่า GUI
3. **Num Envs**: Sweet spot ที่ ~4096 สำหรับ humanoid
4. **Network Size**: เครือข่ายใหญ่ช้ากว่า

---

## เอกสารอ้างอิง

- **PPO Algorithm**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **RSL-RL**: [Repository](https://github.com/leggedrobotics/rsl_rl)
- **Isaac Lab Docs**: [Documentation](https://isaac-sim.github.io/IsaacLab/)

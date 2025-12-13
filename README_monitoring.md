# Isaac Lab Training Monitoring Tools

เครื่องมือสำหรับการวิเคราะห์และติดตามประสิทธิภาพการเทรน Isaac Lab รวมถึงการใช้ GPU, CPU, RAM และข้อมูลต่างๆ

## เครื่องมือที่มี

### 1. `training_analysis.py` - เครื่องมือวิเคราะห์แบบครบถ้วน

**ความสามารถ:**
- ติดตาม GPU usage, memory, temperature
- ติดตาม CPU และ RAM usage
- สร้าง real-time dashboard
- บันทึกข้อมูลเป็น JSON
- แสดงสถิติการเทรนแบบ real-time

**การใช้งาน:**

```bash
# แสดงสถิติปัจจุบัน
python3 training_analysis.py --mode stats

# ติดตามแบบ console (แสดงใน terminal)
python3 training_analysis.py --mode monitor

# ติดตามแบบ dashboard (GUI แสดงกราฟ)
python3 training_analysis.py --mode dashboard

# ติดตามและบันทึกไฟล์
python3 training_analysis.py --mode monitor --save

# กำหนดเวลาการอัปเดต
python3 training_analysis.py --mode monitor --interval 2.0

# กำหนดระยะเวลาการติดตาม (วินาที)
python3 training_analysis.py --mode monitor --duration 300
```

### 2. `training_monitor_wrapper.py` - Wrapper สำหรับรัน training พร้อม monitoring

รัน Isaac Lab training พร้อมกับติดตามระบบอัตโนมัติ

```bash
# เทรน Cartpole พร้อม monitoring
python3 training_monitor_wrapper.py --task Isaac-Cartpole-v0 --framework rsl_rl

# เทรน Humanoid AMP พร้อม monitoring
python3 training_monitor_wrapper.py --task Isaac-Humanoid-Direct-v0 --framework rsl_rl

# เทรนแบบ headless พร้อม dashboard
python3 training_monitor_wrapper.py --task Isaac-Velocity-Flat-Anymal-C-v0 --framework rsl_rl --headless --monitor-mode dashboard
```

### 3. `quick_gpu_check.py` - เช็คสถานะระบบอย่างรวดเร็ว

เช็คสถานะ GPU, CPU, RAM, Disk แบบรวดเร็ว

```bash
# เช็คครั้งเดียว
python3 quick_gpu_check.py

# ติดตามแบบต่อเนื่อง (อัปเดตทุก 5 วินาที)
python3 quick_gpu_check.py --continuous

# ติดตามทุก 2 วินาที เป็นเวลา 60 วินาที
python3 quick_gpu_check.py -c -i 2 -d 60
```

## ตัวอย่างผลลัพธ์

### สถิติการเทรน
```
============================================================
ISAAC LAB TRAINING ANALYSIS
============================================================
Training Duration: 00:15:32
Timestamp: 2025-08-11 10:15:32

CPU Usage: 85.2%
RAM Usage: 12.8GB / 62.4GB (20.5%)

GPU: NVIDIA RTX A2000 12GB
GPU Usage: 95.3%
GPU Memory: 8524MB / 11514MB (74.0%)
GPU Temperature: 78°C

Training Episodes: 150
Average Reward: 245.67
Latest Reward: 312.45
```

### การใช้ GPU ขณะเทรน
- **GPU Usage**: แสดงเปอร์เซ็นต์การใช้งาน GPU
- **GPU Memory**: แสดงหน่วยความจำที่ใช้และเปอร์เซ็นต์
- **Temperature**: อุณหภูมิ GPU (ควรไม่เกิน 85°C)

### ข้อมูลที่บันทึก
- ประวัติการใช้ GPU, CPU, RAM
- เวลาการเทรนและจำนวน episode
- Rewards และ metrics ต่างๆ
- บันทึกเป็น JSON ไฟล์

## การใช้งานที่แนะนำ

### สำหรับการเทรนทั่วไป
```bash
python3 training_monitor_wrapper.py --task [TASK_NAME] --framework rsl_rl --save-stats
```

### สำหรับการ debug หรือเช็คระบบ
```bash
python3 quick_gpu_check.py --continuous --interval 1
```

### สำหรับการวิเคราะห์แบบละเอียด
```bash
python3 training_analysis.py --mode dashboard --save
```

## ไฟล์ที่บันทึก

เครื่องมือจะสร้างไฟล์:
- `training_stats_YYYYMMDD_HHMMSS.json` - ข้อมูลสถิติการติดตาม
- ประกอบด้วย: timestamps, GPU/CPU/RAM history, training metrics

## ข้อกำหนดระบบ

- Python 3.10+
- NVIDIA GPU พร้อม drivers
- Isaac Lab environment
- Dependencies: `GPUtil`, `psutil`, `matplotlib`, `numpy`

## การติดตั้ง Dependencies

```bash
pip install GPUtil matplotlib psutil numpy
```

## หมายเหตุ

- GPU temperature ปกติควรอยู่ที่ 40-80°C
- GPU usage สูง (80-100%) เป็นสิ่งที่ดีสำหรับการเทรน
- RAM usage ขึ้นอยู่กับขนาดของ environment และ batch size
- สามารถปรับ update interval ได้ตามความเหมาะสม
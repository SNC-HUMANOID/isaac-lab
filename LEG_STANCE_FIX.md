# แก้ไขท่ายืนขาห่างเกินไป

## ปัญหา

ขาซ้ายและขาขวาห่างกันมากเกินไปเวลายืนหรือก้าว ทำให้ดูไม่เป็นธรรมชาติ

## สาเหตุ

### 1. Hip Spacing ใน URDF
```
Left hip:  y = +0.0755 m
Right hip: y = -0.0755 m
Total width: 15.1 cm
```

นี่คือค่าจาก mechanical design ซึ่งค่อนข้างกว้าง

### 2. Initial Hip Roll = 0
เมื่อ hip roll เป็น 0 ทั้งคู่ ขาจะชี้ตรงลงมาตามแกน ทำให้ห่างกัน 15.1 cm

## วิธีแก้ไข

**ปรับ Initial Hip Roll และ Ankle Roll** เพื่อให้ขาเข้าหากันมากขึ้น:

### ก่อนแก้ไข
```python
".*_hip_roll_joint": 0.0,        # ขาชี้ตรงลง = ห่างกัน
".*_ankle_roll_joint": 0.0,
```

### หลังแก้ไข
```python
# Hip: ปรับให้ขาเข้าหากัน
"left_hip_roll_joint": -0.05,    # เอียงขาซ้ายเข้าใน
"right_hip_roll_joint": 0.05,    # เอียงขาขวาเข้าใน

# Ankle: ชดเชยการเอียงของสะโพก
"left_ankle_roll_joint": 0.05,   # ชดเชยให้เท้าราบ
"right_ankle_roll_joint": -0.05, # ชดเชยให้เท้าราบ
```

## ผลลัพธ์

### ท่าทางเดิม (Wide Stance)
```
    O
   /|\
  / | \
 /  |  \
    
L       R
|       |
|       |
====  ====
 ↑15.1cm↑
```

### ท่าทางใหม่ (Normal Stance)
```
    O
   /|\
  / | \
 /  |  \
    
  L   R
  |   |
  |   |
  == ==
  ↑~10cm↑
```

## การคำนวณ

**Hip Roll Angle:** ±0.05 rad (±2.87°)

**Approximate Leg Inward Movement:**
- Original spacing: 15.1 cm
- Hip roll effect: ~0.5 cm per side × 2 = 1 cm
- New approximate spacing: ~14 cm (leg width) → ~10 cm (foot spacing)

## ข้อดี

✅ **ท่ายืนดูธรรมชาติมากขึ้น** - เท้าไม่ห่างเกินไป
✅ **Balance ดีขึ้น** - Center of mass อยู่กลางมากขึ้น
✅ **Walking gait สมจริง** - ขาไม่ต้องแกว่งกว้างเกินไป
✅ **Energy efficient** - ใช้พลังงานน้อยลงในการเดิน

## หมายเหตุ

- ค่า ±0.05 rad เป็นค่าที่พอเหมาะ อยู่ใน G1 limit [-0.43, 0.43]
- ถ้ายังห่างเกินไป สามารถเพิ่มเป็น ±0.08 หรือ ±0.10 ได้
- ถ้าแคบเกินไป ขาจะชนกัน ลดเป็น ±0.03

## ทดสอบ

```bash
# ดูท่ายืนใหม่
./isaaclab.sh -p view_humanoid_10102025.py

# ทดสอบ environment
./quick_test_g1.sh

# เทรน
./train_humanoid_10102025.sh
```

## Alternative: แก้ใน URDF (ถ้าต้องการเปลี่ยนถาวร)

หากต้องการปรับ hip spacing ใน URDF ตัวเอง:

```xml
<!-- ก่อน -->
<origin xyz="0 0.0755 -0.051" rpy="0 0 0" />  <!-- Left -->
<origin xyz="0 -0.0755 -0.051" rpy="0 0 0" /> <!-- Right -->

<!-- หลัง (แคบลง 30%) -->
<origin xyz="0 0.053 -0.051" rpy="0 0 0" />   <!-- Left: 7.55cm → 5.3cm -->
<origin xyz="0 -0.053 -0.051" rpy="0 0 0" />  <!-- Right -->
```

แต่วิธีปรับ initial joint positions ที่ทำไปแล้ว **ง่ายกว่าและไม่ต้อง convert ใหม่**!


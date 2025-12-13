# Humanoid Robot Training & Deployment Workflow

## Complete System Architecture

```mermaid
flowchart TB
    subgraph Training["🎓 Phase 1: Training in Isaac Sim"]
        A1[Configure Robot<br/>HUMANOID_10102025_CFG] --> A2[Configure Environment<br/>rough_env_cfg.py]
        A2 --> A3[Tune Rewards<br/>stability + walking]
        A3 --> A4[Train with RSL-RL<br/>PPO Algorithm]
        A4 --> A5[Monitor Progress<br/>TensorBoard]
        A5 --> A6{Good Performance?}
        A6 -->|No| A3
        A6 -->|Yes| A7[Save Checkpoint<br/>model_9050.pt]
    end
    
    subgraph Export["📦 Phase 2: Policy Export"]
        B1[Load Checkpoint<br/>model_9050.pt] --> B2[Extract Actor Network<br/>75→512→256→128→21]
        B2 --> B3[Convert to TorchScript<br/>torch.jit.script]
        B3 --> B4[Embed Metadata<br/>obs_dim=75, action_dim=21]
        B4 --> B5[Save Deployment Model<br/>humanoid_policy.pt]
    end
    
    subgraph Validate["✅ Phase 3: Validation"]
        C1[Test Policy Loading<br/>torch.jit.load] --> C2[Verify Dimensions<br/>input=75, output=21]
        C2 --> C3[Run Simulation Test<br/>test_policy_simulation.py]
        C3 --> C4{Actions Valid?}
        C4 -->|No| Export
        C4 -->|Yes| C5[Ready for Deployment]
    end
    
    subgraph Deploy["🤖 Phase 4: Real Robot Deployment"]
        D1[Configure Robot<br/>joint names, limits, defaults] --> D2[Copy Files to Robot<br/>policy.pt + node.py]
        D2 --> D3[Install Dependencies<br/>PyTorch, ROS 2]
        D3 --> D4[Launch ROS 2 Node<br/>humanoid_policy_node]
        D4 --> D5[Subscribe to Sensors<br/>joints + IMU]
        D5 --> D6[Build Observations<br/>75 dimensions]
        D6 --> D7[Run Policy Inference<br/>@200Hz]
        D7 --> D8[Publish Joint Commands<br/>21 positions]
        D8 --> D9{Safe Operation?}
        D9 -->|No| D10[Emergency Stop]
        D9 -->|Yes| D11[Continue Control]
        D11 --> D5
    end
    
    A7 --> B1
    B5 --> C1
    C5 --> D1
    D10 --> D12[Adjust Parameters<br/>action_scale, filters]
    D12 --> D4
    
    style Training fill:#e1f5ff
    style Export fill:#fff4e1
    style Validate fill:#e8f5e9
    style Deploy fill:#fce4ec
```

## Data Flow Architecture

```mermaid
graph LR
    subgraph Robot["🤖 Real Robot Hardware"]
        J[Joint Encoders<br/>21 positions + velocities] 
        I[IMU Sensor<br/>acceleration + gyroscope]
        M[Motor Controllers<br/>21 actuators]
    end
    
    subgraph ROS2["🔄 ROS 2 Middleware"]
        JS[/joint_states<br/>sensor_msgs/JointState]
        IMU[/imu/data<br/>sensor_msgs/Imu]
        CMD[/joint_position_command<br/>Float32MultiArray]
        VEL[/cmd_vel<br/>geometry_msgs/Twist]
    end
    
    subgraph Node["🧠 Policy Node"]
        OB[Observation Builder<br/>75 dimensions]
        POL[Policy Network<br/>TorchScript]
        ACT[Action Processor<br/>safety + scaling]
    end
    
    J -->|publish| JS
    I -->|publish| IMU
    JS --> OB
    IMU --> OB
    VEL -->|user commands| OB
    
    OB -->|obs_tensor| POL
    POL -->|action_tensor| ACT
    ACT -->|publish| CMD
    CMD -->|subscribe| M
    
    style Robot fill:#ffebee
    style ROS2 fill:#e3f2fd
    style Node fill:#f3e5f5
```

## Observation Structure (75 Dimensions)

```mermaid
graph TD
    subgraph Obs["📊 Observation Vector [75]"]
        O1[1-3: Base Linear Velocity<br/>vx, vy, vz]
        O2[4-6: Base Angular Velocity<br/>ωx, ωy, ωz]
        O3[7-9: Projected Gravity<br/>gx, gy, gz normalized]
        O4[10-12: Velocity Commands<br/>cmd_vx, cmd_vy, cmd_ω]
        O5[13-33: Joint Positions Relative<br/>pos - default_pos]
        O6[34-54: Joint Velocities<br/>21 joint velocities]
        O7[55-75: Previous Actions<br/>last policy output]
    end
    
    JS[/joint_states] --> O5
    JS --> O6
    IMU[/imu/data] --> O2
    IMU --> O3
    VEL[/cmd_vel] --> O4
    HIST[Action History] --> O7
    
    O1 & O2 & O3 & O4 & O5 & O6 & O7 --> POL[Policy Network]
    
    style Obs fill:#fff9c4
```

## Policy Network Architecture

```mermaid
graph LR
    subgraph Network["🧠 Actor Network"]
        I[Input<br/>75 dims] --> L1[Linear + ELU<br/>75 → 512]
        L1 --> L2[Linear + ELU<br/>512 → 256]
        L2 --> L3[Linear + ELU<br/>256 → 128]
        L3 --> L4[Linear<br/>128 → 21]
        L4 --> O[Output<br/>21 actions]
    end
    
    O --> S[Scale × 0.25]
    S --> ADD[Add to Default Pose]
    ADD --> CLIP[Clip to Joint Limits]
    CLIP --> SAFE[Velocity Limiter]
    SAFE --> CMD[Joint Commands]
    
    style Network fill:#e1bee7
    style CMD fill:#c8e6c9
```

## Control Loop Timing

```mermaid
sequenceDiagram
    participant Robot
    participant ROS2
    participant Node
    participant Policy
    
    Note over Node: Control Loop @ 200Hz (5ms)
    
    loop Every 5ms
        Robot->>ROS2: Publish /joint_states
        Robot->>ROS2: Publish /imu/data
        ROS2->>Node: Receive sensor data
        
        Node->>Node: Build observation (75D)
        Node->>Policy: Forward pass
        Policy-->>Node: Action (21D)
        
        Node->>Node: Apply safety filters
        Node->>Node: Scale & clip action
        Node->>ROS2: Publish /joint_position_command
        ROS2->>Robot: Send to motor controllers
    end
    
    Note over Node,Robot: Total latency < 5ms
```

## Deployment Sequence

```mermaid
sequenceDiagram
    participant User
    participant Sim as Isaac Sim
    participant Export as Export Script
    participant Robot as Real Robot
    participant ROS as ROS 2 Node
    
    User->>Sim: 1. Train policy
    Sim-->>User: model_9050.pt
    
    User->>Export: 2. Run export script
    Export->>Export: Load checkpoint
    Export->>Export: Extract actor
    Export->>Export: Convert to TorchScript
    Export-->>User: humanoid_policy.pt
    
    User->>User: 3. Validate with test script
    
    User->>Robot: 4. Copy files (scp)
    User->>Robot: 5. Configure robot params
    
    User->>ROS: 6. Launch policy node
    ROS->>Robot: Subscribe to sensors
    Robot-->>ROS: Stream sensor data
    
    loop Control Loop
        ROS->>ROS: Build observation
        ROS->>ROS: Run inference
        ROS->>Robot: Send commands
    end
```

## Safety System

```mermaid
graph TD
    A[Policy Output] --> B{Action Magnitude<br/>Check}
    B -->|> 1.0| C[Clip to ±1.0]
    B -->|OK| D{Moving Average<br/>Filter}
    C --> D
    
    D --> E[Scale by action_scale]
    E --> F{Joint Limits<br/>Check}
    F -->|Violates| G[Clip to Limits]
    F -->|OK| H{Velocity Limit<br/>Check}
    G --> H
    
    H -->|Too Fast| I[Limit Change Rate]
    H -->|OK| J[Send to Motors]
    I --> J
    
    K[Emergency Stop] -.->|Override| J
    
    style K fill:#ff5252,color:#fff
    style J fill:#4caf50,color:#fff
```

## File Structure

```mermaid
graph TD
    subgraph IsaacLab["📁 IsaacLab/"]
        A[source/isaaclab_tasks/.../rough_env_cfg.py<br/>Environment & Rewards]
        B[logs/rsl_rl/.../model_9050.pt<br/>Training Checkpoint]
        
        subgraph ROS["📁 ros2/"]
            C[export_policy_torchscript.py<br/>Export Script]
            D[humanoid_policy.pt<br/>Deployed Model]
            E[humanoid_policy_node.py<br/>ROS 2 Control Node]
            F[test_policy_simulation.py<br/>Validation Script]
            G[robot_config.yaml<br/>Robot Configuration]
            H[README.md<br/>Documentation]
        end
    end
    
    A -.->|train| B
    B -->|export| C
    C -->|creates| D
    D & E & G -->|deploy| I[Real Robot]
    F -.->|validate| D
    
    style A fill:#bbdefb
    style B fill:#c5cae9
    style D fill:#c8e6c9
    style E fill:#fff9c4
    style I fill:#ffccbc
```

## Troubleshooting Flow

```mermaid
graph TD
    START[Robot Issue] --> Q1{Policy loads?}
    Q1 -->|No| F1[Check PyTorch version<br/>Verify file path]
    Q1 -->|Yes| Q2{Correct dimensions?}
    
    Q2 -->|No| F2[Check observation builder<br/>Must be 75D input]
    Q2 -->|Yes| Q3{Actions reasonable?}
    
    Q3 -->|No| F3[Check action_scale<br/>Start with 0.05-0.1]
    Q3 -->|Yes| Q4{Robot stable?}
    
    Q4 -->|No| F4[Reduce action_scale<br/>Add filtering<br/>Check IMU frame]
    Q4 -->|Yes| Q5{Follows commands?}
    
    Q5 -->|No| F5[Verify /cmd_vel topic<br/>Check command scaling]
    Q5 -->|Yes| SUCCESS[✓ Working!]
    
    F1 & F2 & F3 & F4 & F5 --> RETRY[Retry Test]
    RETRY --> START
    
    style SUCCESS fill:#4caf50,color:#fff
    style START fill:#ff9800,color:#fff
```


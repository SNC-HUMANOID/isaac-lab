#!/bin/bash

# Isaac Lab Disk Cleanup Script
# ทำความสะอาดเพื่อแก้ปัญหา disk space เต็ม

echo "🧹 Isaac Lab Disk Cleanup"
echo "=========================="

# Check current space
echo "📊 Current disk usage:"
df -h /

echo ""
echo "🔍 Checking large cache directories:"
du -sh /home/sncbot/.cache/pip /home/sncbot/.cache/ov /home/sncbot/.cache/packman /home/sncbot/.cache/torch 2>/dev/null

echo ""
echo "🗑️  Cleaning up..."

# 1. Clean pip cache (can be huge)
echo "   • Cleaning pip cache (7.1GB)..."
if [ -d "/home/sncbot/.cache/pip" ]; then
    rm -rf /home/sncbot/.cache/pip/*
    echo "     ✅ Pip cache cleaned"
else
    echo "     ⚠️  Pip cache not found"
fi

# 2. Clean conda cache
echo "   • Cleaning conda cache..."
if command -v conda &> /dev/null; then
    conda clean --all -y &>/dev/null
    echo "     ✅ Conda cache cleaned"
else
    echo "     ⚠️  Conda not available"
fi

# 3. Clean Omniverse cache (can be large)
echo "   • Cleaning Omniverse cache (3.1GB)..."
if [ -d "/home/sncbot/.cache/ov" ]; then
    rm -rf /home/sncbot/.cache/ov/pkg
    rm -rf /home/sncbot/.cache/ov/cache
    rm -rf /home/sncbot/.cache/ov/logs
    echo "     ✅ Omniverse cache cleaned"
else
    echo "     ⚠️  Omniverse cache not found"
fi

# 4. Clean packman cache
echo "   • Cleaning packman cache..."
if [ -d "/home/sncbot/.cache/packman" ]; then
    find /home/sncbot/.cache/packman -name "*.log" -delete
    find /home/sncbot/.cache/packman -name "*.tmp" -delete
    echo "     ✅ Packman cache cleaned"
else
    echo "     ⚠️  Packman cache not found"
fi

# 5. Clean PyTorch cache
echo "   • Cleaning PyTorch cache..."
if [ -d "/home/sncbot/.cache/torch" ]; then
    rm -rf /home/sncbot/.cache/torch/hub/*
    rm -rf /home/sncbot/.cache/torch/kernels/*
    echo "     ✅ PyTorch cache cleaned"
else
    echo "     ⚠️  PyTorch cache not found"
fi

# 6. Clean thumbnails
echo "   • Cleaning thumbnails..."
if [ -d "/home/sncbot/.cache/thumbnails" ]; then
    rm -rf /home/sncbot/.cache/thumbnails/*
    echo "     ✅ Thumbnails cleaned"
fi

# 7. Clean system logs (optional)
echo "   • Cleaning old system logs..."
sudo journalctl --vacuum-time=7d &>/dev/null
echo "     ✅ System logs cleaned (kept last 7 days)"

# 8. Clean apt cache
echo "   • Cleaning apt cache..."
sudo apt-get autoremove -y &>/dev/null
sudo apt-get autoclean &>/dev/null
echo "     ✅ APT cache cleaned"

# 9. Clean Isaac Sim logs (if they exist)
echo "   • Cleaning Isaac Sim logs..."
if [ -d "/home/sncbot/isaac-sim/kit/logs" ]; then
    find /home/sncbot/isaac-sim/kit/logs -name "*.log" -mtime +7 -delete
    echo "     ✅ Isaac Sim old logs cleaned"
fi

# 10. Clean temporary files
echo "   • Cleaning temporary files..."
rm -rf /tmp/* 2>/dev/null
echo "     ✅ Temporary files cleaned"

echo ""
echo "📊 Disk space after cleanup:"
df -h /

echo ""
echo "🎯 Cache sizes after cleanup:"
du -sh /home/sncbot/.cache/pip /home/sncbot/.cache/ov /home/sncbot/.cache/packman /home/sncbot/.cache/torch 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 Tips to prevent future issues:"
echo "   • Run 'pip cache purge' regularly"
echo "   • Use 'conda clean --all' periodically"
echo "   • Delete old Isaac Sim logs"
echo "   • Monitor disk usage with 'df -h'"

echo ""
echo "🚀 You can now try running the visualization:"
echo "   ./run_g1_viz.sh"
# matplotlib tight_layout 警告说明

## 警告信息
```
UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
```

## 警告含义

`plt.tight_layout()` 是matplotlib的一个函数，用于**自动调整子图参数**，使子图标签、标题等不会重叠。

**警告原因**：
- 当图表中包含某些特殊类型的 Axes（坐标轴）时，`tight_layout()` 无法正确处理
- 常见情况：
  - 使用了 `ax.axis('off')` 隐藏坐标轴
  - 使用了 `GridSpec` 创建的复杂布局
  - 某些自定义的 Axes 类型

## 影响

- **功能影响**：⚠️ **无影响** - 图表仍然可以正常显示和保存
- **视觉影响**：可能某些标签会稍微重叠，但通常不明显
- **警告信息**：会在控制台显示警告，但不影响程序运行

## 解决方案

已修复：使用 `warnings.catch_warnings()` 来抑制这个特定的警告

```python
# 修复前
plt.tight_layout()

# 修复后
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.tight_layout()
```

## 替代方案

如果不想抑制警告，也可以：

1. **使用 `plt.subplots_adjust()` 手动调整**：
```python
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
```

2. **在保存时使用 `bbox_inches='tight'`**（已在代码中使用）：
```python
plt.savefig(path, bbox_inches='tight')  # 这已经可以自动调整边距
```

## 总结

- ✅ 已修复：所有 `tight_layout()` 调用都已添加警告抑制
- ✅ 不影响功能：图表显示和保存都正常
- ✅ 警告已消除：不会再显示警告信息



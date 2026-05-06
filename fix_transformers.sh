#!/bin/bash
# 修复 transformers API 兼容性问题
# 上传到服务器后执行: bash /root/MIRA/fix_transformers.sh

cd /root/MIRA

# 清理 pyc 缓存
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

python3 << 'PYEOF'
p = 'mira/models/modeling_mira.py'
text = open(p).read()

# Fix 1: from_legacy_cache
text = text.replace(
    'past_key_values = DynamicCache.from_legacy_cache(past_key_values)',
    'past_key_values = DynamicCache()'
)

# Fix 2: get_usable_length -> get_seq_length (所有变体)
text = text.replace('.get_usable_length(seq_length)', '.get_seq_length()')
text = text.replace('.get_usable_length(None, self.layer_idx)', '.get_seq_length()')
text = text.replace('.get_usable_length(kv_seq_len, self.layer_idx)', '.get_seq_length()')

open(p, 'w').write(text)
print('fix_transformers.sh 执行完毕，共修复 5 处 API 调用')
PYEOF

echo "修复完成，可重新运行评估脚本"

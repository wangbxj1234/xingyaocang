---
license: other
task_categories:
- video-text-to-text
language:
- en
tags:
- robotics
pretty_name: 复核包装发货循环
size_categories:
- 1K<n<10K
---

# 复核包装发货循环

每条都是桌上的一单：扫码、装进外包装、贴面单。外包装是套进袋子、气泡膜，或者拉胶带封盒。

片子从上一张面单之后开始，到这一张面单贴完。`cycles.jsonl` 里的 `captions` 时间相对这条片子的开头。`video` 指向 `videos/` 里的 mp4。

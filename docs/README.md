# Safe MARL 论文展示页

这是论文 **Safe Multi-Agent Reinforcement Learning for Robust and Scalable Multi-Vehicle Control in Cooperative Transportation Systems** 的独立英文项目网页。标题与 Abstract 参考 `../LGDRL-page/`，整体采用 [ProbeGen](https://vision.huji.ac.il/probegen/) 的简洁学术页面风格。纯 HTML / CSS / JavaScript，无构建步骤、无第三方字体或 CDN 依赖。当前目录的内容可作为 GitHub Pages 站点根目录。

## 展示内容

1. 论文标题、作者、单位、论文 PDF、代码入口与引用。
2. 直接展开的 Abstract。
3. Framework：框架图、简短图注及 01 / 02 / 03 三个组成部分。
4. Experiment 1：简短实验说明、Ours / IPPO / MADDPG / MAPPO-Lag 四个视频、主测试表。
5. Experiment 2：简短实验说明、车辆数量 / 载荷形状选项卡，以及跟随选项卡切换的视频和结果摘要。
6. 可复制、可下载的 BibTeX。

标题、摘要、方法说明、图片、实验结果及下载的 PDF 已对齐当前 `paper.tex` / `paper.pdf`；网页中的数值是静态内容，论文更新后需要同步修改。论文副本位于 `assets/paper/SSR-MARL.pdf`。

## 本地查看

可以直接打开 `index.html`。推荐在此目录启动静态服务，以检查与部署时相同的相对路径：

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

访问 `http://127.0.0.1:8000/`。没有 Python 时可使用编辑器的静态网页预览功能。

## 补充视频

以下目录已创建，`.gitkeep` 用于让 Git 保留空文件夹。按下表命名并放入 MP4，刷新页面即可自动替换对应占位，无须改 HTML。

| 位置 | 视频路径（相对于本目录） | 建议内容 |
| --- | --- | --- |
| 完整方法 | `assets/videos/comparison/ssr-marl.mp4` | 完整方法在选定任务中的运输过程 |
| IPPO | `assets/videos/comparison/ippo.mp4` | 与完整方法使用相同配置、路径和播放速度 |
| MADDPG | `assets/videos/comparison/maddpg.mp4` | 与完整方法使用相同配置、路径和播放速度 |
| MAPPO-Lag | `assets/videos/comparison/mappo-lag.mp4` | 与完整方法使用相同配置、路径和播放速度 |
| 车辆数量 | `assets/videos/scalability/vehicle-counts.mp4` | 四、五、六车运输矩形载荷；标注同一策略、无重新训练 |
| 载荷形状 | `assets/videos/scalability/payload-shapes.mp4` | 五边形、T 形、L 形载荷，展示未见路径 |

推荐 MP4 / H.264、16:9、720p 或 1080p。播放器保留原始宽高比，不裁剪画面；有解说时可提供 WebVTT 字幕。

配置位于 `assets/js/config.js`：

```javascript
ours: {
  src: "assets/videos/comparison/ssr-marl.mp4",
  poster: "assets/images/ssr-marl-poster.jpg", // 可选；留空也可播放
  captions: "assets/videos/comparison/ssr-marl.en.vtt" // 可选
}
```

视频不会自动播放。实验一的四个播放器可以独立播放，方便并排比较。实验二切换选项卡时同步切换视频与结果，隐藏的视频自动暂停，再次切回保留播放进度。尚未上传、路径错误或编码不支持时，保留占位。浏览器仅在接近相应区块时请求视频；缺失文件可能在开发者工具中显示 404，这是当前预留素材的预期状态。把配置中的 `src` 改为 `""` 可完全停止对应请求。

当前页面共使用六个视频位置。原先 overview、robustness、ablation 的目录和配置保留，但当前页面不显示或请求它们。

## 其他待补项

- **代码地址**：在 `assets/js/config.js` 设置 `codeUrl` 后自动显示 Code 按钮。当前留空，页面显示 “Code · coming soon”。
- **论文 PDF**：用最新版本覆盖 `assets/paper/SSR-MARL.pdf`。
- **正式出版信息**：目前没有填写期刊、会议或 DOI；`assets/citation.bib` 与 `index.html` 内的引用暂用 manuscript。正式出版后同时更新这两处。
- **站点地址**：按论文给出的 `https://bitmobility.github.io/SSR-MARL/` 填入引用与 Open Graph 图片地址；如果地址调整，请在 `index.html` 和 `assets/citation.bib` 中同步修改。

## GitHub Pages 发布

本目录本身就是网站根目录。将其**内部内容**（包括 `index.html`、`assets/` 和 `.nojekyll`）放到计划用于展示论文的仓库根目录，再选择该仓库的 GitHub Pages 分支根目录作为来源。所有页面资源均使用相对路径，支持 `/SSR-MARL/` 项目子路径。

如果保留仓库中的 `git_page/` 子目录，则需要使用自己的 Pages 工作流上传该目录；不要将包含论文源文件的整个工作区作为网站发布。这里仅生成网页文件，尚未上传或发布。

## 文件维护

```text
git_page/
├── index.html                 页面文案、表格、结构
├── .nojekyll                  静态站点标记
├── README.md                  本说明
└── assets/
    ├── css/style.css          响应式样式
    ├── js/config.js           视频路径、代码仓库地址
    ├── js/main.js             视频、页签、图示放大、复制引用
    ├── images/                论文原图的 WebP 副本及已有素材
    ├── paper/SSR-MARL.pdf     论文副本
    ├── citation.bib           引用下载
    └── videos/                按用途预留的视频目录
```

论文图表保持原始内容，转换为无损 WebP 用于网页。作者、单位、指标、模型名称与实验结论均以本论文为准。页面采用白底标题区与浅灰 Abstract 区，不再使用照片背景。

## 与论文同步

网页不再使用 SSR-MARL 作为方法名称，实验中统一显示 Ours。现有站点地址、PDF 文件名和视频路径保留，以兼容已有链接和后续素材。

框架图和六张构型图使用论文原始 PNG；更新论文图片时，请同步 `assets/images/` 下对应文件。摘要与论文一致，仅省略指向本网页的末尾链接句。

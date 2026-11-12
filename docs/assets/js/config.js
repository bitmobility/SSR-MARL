// All paths are relative to index.html. Drop MP4 files at these paths;
// the page automatically replaces placeholders when the videos are available.
// Use src: "" to keep a placeholder without requesting a file.
// Optional poster and captions paths can be supplied for each video.
// Current page: four comparison videos + two tab-dependent scalability videos.
// overview, robustness, and ablation are retained for later use, not displayed.
window.PROJECT_CONFIG = {
  codeUrl: "https://github.com/bitmobility/SSR-MARL", // Set the public repository URL when it is ready.
  videos: {
    overview: { src: "assets/videos/overview/overview.mp4", poster: "" },
    ours: { src: "assets/videos/comparison/ssr-marl.mp4", poster: "assets/videos/comparison/ssr-marl.jpg" },
    ippo: { src: "assets/videos/comparison/ippo.mp4?v=2", poster: "assets/videos/comparison/ippo.jpg" },
    maddpg: { src: "assets/videos/comparison/maddpg.mp4", poster: "assets/videos/comparison/maddpg.jpg" },
    "mappo-lag": { src: "assets/videos/comparison/mappo-lag.mp4", poster: "assets/videos/comparison/mappo-lag.jpg" },
    robustness: { src: "assets/videos/robustness/model-error-sweep.mp4", poster: "" },
    "vehicle-counts": { src: "assets/videos/scalability/vehicle-counts.mp4?v=3", poster: "assets/videos/scalability/vehicle-counts.jpg" },
    "payload-shapes": { src: "assets/videos/scalability/payload-shapes.mp4?v=3", poster: "assets/videos/scalability/payload-shapes.jpg" },
    ablation: { src: "assets/videos/ablation/component-comparison.mp4", poster: "" }
  }
};

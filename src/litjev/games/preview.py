"""Local, browser-paced Doom preview. Uniform random actions; no model or GPU."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from litjev.games import make_env
from litjev.vision import encode_image


class RandomPreview:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        self.episode = 0
        self.rng = np.random.default_rng(seed)
        self.lock = Lock()
        self.reset()

    def reset(self):
        self.frame, _ = self.env.reset(seed=self.seed + self.episode)
        self.episode += 1
        self.step = 0
        self.reward = self.total_reward = 0.0
        self.done = False
        self.action = None
        self.step_ms = 0.0

    def advance(self):
        if self.done:
            self.reset()
            return
        started = perf_counter()
        action = int(self.rng.integers(self.env.action_space.n))
        self.frame, self.reward, terminated, truncated, _ = self.env.step(action)
        self.step_ms = (perf_counter() - started) * 1000
        self.total_reward += self.reward
        self.step += 1
        self.action = self.env.unwrapped.action_names[action]
        self.done = terminated or truncated

    def snapshot(self):
        actions = self.env.unwrapped.action_names
        return {
            "policy": "uniform_random",
            "forward_calls": 0,
            "frame": encode_image(self.frame),
            "actions": actions,
            "probabilities": [1 / len(actions)] * len(actions),
            "episode": self.episode,
            "step": self.step,
            "action": self.action,
            "reward": self.reward,
            "total_reward": self.total_reward,
            "done": self.done,
            "step_ms": self.step_ms,
        }


def create_preview_app(seed=7, max_steps=1000, resolution="640x480"):
    @asynccontextmanager
    async def lifespan(app):
        with make_env("doom", max_steps=max_steps, resolution=resolution) as env:
            app.state.preview = RandomPreview(env, seed)
            yield

    app = FastAPI(title="LitJev Doom — Random Preview", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index():
        template = (Path(__file__).parents[1] / "static" / "film.html").read_text()
        return HTMLResponse(template.replace('"__LITJEV_LIVE__"', 'true', 1))

    @app.get("/state")
    def state():
        preview = app.state.preview
        with preview.lock:
            return preview.snapshot()

    @app.post("/step")
    def step():
        preview = app.state.preview
        with preview.lock:
            preview.advance()
            return preview.snapshot()

    @app.post("/reset")
    def reset():
        preview = app.state.preview
        with preview.lock:
            preview.reset()
            return preview.snapshot()

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    uvicorn.run(create_preview_app(seed=args.seed), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

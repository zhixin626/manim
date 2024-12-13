#!/usr/bin/env python
from addict import Dict

from manimlib import __version__
from manimlib.config import manim_config
from manimlib.config import parse_cli
import manimlib.extract_scene
from manimlib.utils.cache import clear_cache
from manimlib.window import Window


from IPython.terminal.embed import KillEmbedded


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from argparse import Namespace


def run_scenes():
    """
    Runs the scenes in a loop and detects when a scene reload is requested.
    """
    # Create a new dict to be able to upate without
    # altering global configuration
    scene_config = Dict(manim_config.scene)
    run_config = manim_config.run

    if run_config.show_in_window:
        # Create a reusable window
        window = Window(**manim_config.window)
        scene_config.update(window=window)

    while True:
        try:
            # Blocking call since a scene may init an IPython shell()
            scenes = manimlib.extract_scene.main(scene_config, run_config)
            for scene in scenes:
                scene.run()
            return
        except KillEmbedded:
            # Requested via the `exit_raise` IPython runline magic
            # by means of the reload_scene() command
            pass
        except KeyboardInterrupt:
            break


def main():
    """
    Main entry point for ManimGL.
    """
    print(f"ManimGL \033[32mv{__version__}\033[0m")

    args = parse_cli()
    if args.version and args.file is None:
        return
    if args.clear_cache:
        clear_cache()

    run_scenes()


if __name__ == "__main__":
    main()

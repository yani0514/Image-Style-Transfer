from _bootstrap import bootstrap

bootstrap()

from style_transfer.cli import main


if __name__ == "__main__":
    main(["adain", *(__import__("sys").argv[1:])])

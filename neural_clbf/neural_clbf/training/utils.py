import subprocess


def current_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
            )
    except:
        return "NO COMMIT HASH"

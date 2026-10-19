#!/bin/bash
# Install Scrivo on a Mac with one line:
#
#   curl -fsSL https://raw.githubusercontent.com/zohartito/scrivo/main/scripts/get.sh | bash
#
# Downloads Scrivo with git into ~/scrivo (or $SCRIVO_SOURCE; a copy installed
# before the rename, in ~/sotto, stays there) and runs its scripts/install-mac.sh, or
# updates that copy with install-mac.sh --update (the new packages first, the
# source only after them), then opens Scrivo. Nothing needs sudo. Missing
# tools are named with the command that installs them.
# Everything sits inside main(), so a partly downloaded script never runs.
set -euo pipefail

# Same repository? Ignores https vs either ssh form (git@github.com:… and
# ssh://git@github.com/…), a trailing .git and a trailing slash.
same_repo() {
    local normalized=()
    local url
    for url in "$1" "$2"; do
        url="${url/#git@github.com:/https://github.com/}"
        url="${url/#ssh:\/\/git@github.com\//https://github.com/}"
        url="${url%/}"
        normalized+=("${url%.git}")
    done
    [ "${normalized[0]}" = "${normalized[1]}" ]
}

# The first process running the scrivo.py (or sotto.py, its name before the
# rename) in folder $1 (resolved): by that path (what Scrivo.app runs) or
# relative to the process's working directory (a terminal run:
# venv-alpha/bin/python scrivo.py run).
running_from() {
    local listing pid command words word dir
    listing="$(ps -axww -o pid= -o command=)" || return 0
    while read -r pid command; do
        case " $command " in *" $1/scrivo.py "*|*" $1/sotto.py "*) echo "$pid"; return 0 ;; esac
        read -ra words <<< "$command" || true  # split without globbing
        for word in ${words[@]+"${words[@]}"}; do
            case "$word" in scrivo.py|*/scrivo.py|sotto.py|*/sotto.py) ;; *) continue ;; esac
            case "$word" in
                /*) dir="$(dirname "$word")" ;;
                *) dir="$(/usr/sbin/lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')" || continue
                   [ -n "$dir" ] || continue
                   dir="$dir/$(dirname "$word")" ;;
            esac
            dir="$(cd "$dir" 2>/dev/null && pwd -P)" || continue
            [ "$dir" != "$1" ] || { echo "$pid"; return 0; }
        done
    done <<< "$listing"
}

main() {
    # This project, and its URL from before the rename (Scrivo was called
    # Sotto; SOTTO_* is the old name of each SCRIVO_* variable).
    local scrivo_repo=https://github.com/zohartito/scrivo.git
    local legacy_repo=https://github.com/zohartito/sotto.git
    local repo="${SCRIVO_REPO:-${SOTTO_REPO:-$scrivo_repo}}"
    local dest="${SCRIVO_SOURCE:-${SOTTO_SOURCE:-}}"
    local python="${SCRIVO_PYTHON:-${SOTTO_PYTHON:-}}"

    fail() { printf '✗ %s\n' "$*" >&2; exit 1; }
    # A copy of this project: it follows $repo or, for this project, the URL
    # from before the rename (GitHub redirects it).
    follows_this() { same_repo "$1" "$repo" || { same_repo "$repo" "$scrivo_repo" && same_repo "$1" "$legacy_repo"; }; }

    [ "$(uname -s)" = Darwin ] || fail "This installer is for macOS. On Windows, see the README."
    [ "$(uname -m)" = arm64 ] || fail "Scrivo needs an Apple Silicon Mac."
    if ! xcrun --find clang >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        fail "First install Apple's command line tools: xcode-select --install  (then run this line again)"
    fi
    if [ -z "$python" ]; then
        # The first native one: an Intel Homebrew's python3.12 may come first on PATH.
        local candidate found
        for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
            found="$(command -v "$candidate" 2>/dev/null)" || continue
            if [ "$("$found" -c 'import platform; print(platform.machine())' 2>/dev/null)" = arm64 ]; then
                python="$found"
                break
            fi
        done
    fi
    if [ -z "$python" ]; then
        if command -v brew >/dev/null 2>&1; then
            fail "Scrivo needs Python 3.12: brew install python@3.12  (then run this line again)"
        fi
        fail "Scrivo needs Python 3.12: install it from https://www.python.org/downloads/ (then run this line again)"
    fi

    if [ -z "$dest" ]; then
        dest="$HOME/scrivo"
        # A copy installed before the rename is in ~/sotto, and Scrivo.app runs
        # it by that path: update it where it is, unless ~/scrivo exists.
        if [ ! -e "$dest" ] && [ -d "$HOME/sotto/.git" ] \
            && follows_this "$(git -C "$HOME/sotto" remote get-url origin 2>/dev/null || true)"; then
            dest="$HOME/sotto"
        fi
    fi

    local update=""
    if [ -d "$dest/.git" ]; then
        local origin renamed=""
        origin="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
        if ! same_repo "$origin" "$repo"; then
            # A copy made before the rename follows the old URL: the same
            # project, moved to the new URL below.
            follows_this "$origin" \
                || fail "$dest is a git copy of ${origin:-an unknown project}, not Scrivo. Choose another folder with SCRIVO_SOURCE=..."
            renamed="${origin/zohartito\/sotto/zohartito/scrivo}"  # same form: https or ssh
        fi
        # Updating under a running Scrivo would swap its source and packages
        # while they are in use.
        local running
        running="$(running_from "$(cd "$dest" && pwd -P)")"
        [ -z "$running" ] || fail "Scrivo is running from $dest (process $running). Quit it from its menu (or update it there with Check for Updates), then run this line again."
        if [ -n "$renamed" ]; then
            git -C "$dest" remote set-url origin "$renamed"
            echo "== $dest now follows $renamed (Sotto is now called Scrivo)"
        fi
        echo "== Updating Scrivo in $dest"
        # install-mac.sh --update fetches, installs the new packages, and moves
        # the source only after them (putting the packages back on a failure).
        update="--update"
    elif [ -e "$dest" ]; then
        fail "$dest exists and is not a copy of Scrivo. Choose another folder with SCRIVO_SOURCE=..."
    else
        echo "== Downloading Scrivo into $dest"
        git clone --quiet "$repo" "$dest" < /dev/null
    fi

    if [ -n "${SCRIVO_GET_DRY_RUN:-}" ]; then
        echo "dry run: would run $dest/scripts/install-mac.sh ${update:+$update }--python $python"
        return 0
    fi
    # shellcheck disable=SC2086  # $update is one word or nothing
    bash "$dest/scripts/install-mac.sh" $update --python "$python" < /dev/null
    # By its path, where install_app.py built it (/Applications when writable,
    # else ~/Applications): LaunchServices does not know a bundle this new
    # yet, so "open -a Scrivo" found nothing on a fresh Mac.
    local app=/Applications/Scrivo.app
    [ -w /Applications ] || app="$HOME/Applications/Scrivo.app"
    open "$app" || true
}

main "$@"

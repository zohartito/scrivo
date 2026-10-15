#!/bin/bash
# Install, update or remove Scrivo.app (in /Applications, or ~/Applications when that
# is not writable) from this source folder.
#
#   scripts/install-mac.sh              install (or repair) the app
#   scripts/install-mac.sh --update     pull the latest source (git clones), then reinstall
#   scripts/install-mac.sh --uninstall  remove Scrivo.app and its login item; keeps your data
#
# Options: --data-dir DIR (default ~/Library/Application Support/sotto-alpha),
#          --python PATH (a native arm64 Python 3.12).
# Nothing here needs sudo, changes privacy settings or touches another Scrivo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
# Scrivo was called Sotto: SOTTO_* is the old name of each SCRIVO_* variable.
DATA_DIR="${SCRIVO_DATA_DIR:-${SOTTO_DATA_DIR:-$HOME/Library/Application Support/sotto-alpha}}"
PYTHON="${SCRIVO_PYTHON:-${SOTTO_PYTHON:-}}"
MODE=install

while [ $# -gt 0 ]; do
    case "$1" in
        --update) MODE=update ;;
        --uninstall) MODE=uninstall ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --python) PYTHON="$2"; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

fail() { echo "✗ $*" >&2; exit 1; }
VENV="$ROOT/venv-alpha"

# Does this old Sotto.app launcher start this folder's sotto.py (what they ran
# before the module rename)? It holds the path as one NUL-terminated C string
# (scripts/install_app.py): require that exact string, never a substring, like
# mac_app.launcher_runs. No perl: not ours.
launcher_runs_this() {
    command -v perl >/dev/null 2>&1 || return 1
    SCRIPT_PATH="$ROOT/sotto.py" perl -0777 -ne 'exit(index($_, "\0$ENV{SCRIPT_PATH}\0") >= 0 ? 0 : 1)' "$1" 2>/dev/null
}

if [ "$MODE" = uninstall ]; then
    if [ -x "$VENV/bin/python" ]; then
        "$VENV/bin/python" "$ROOT/scripts/install_app.py" --uninstall
    else
        for app in /Applications/Scrivo.app "$HOME/Applications/Scrivo.app"; do
            # Only ever remove a Scrivo.app this installer built.
            if [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist" 2>/dev/null)" = io.github.zohartito.scrivo ]; then
                rm -r "$app"
            fi
        done
        launchctl bootout "gui/$(id -u)/io.github.zohartito.scrivo" 2>/dev/null || true
        rm -f "$HOME/Library/LaunchAgents/io.github.zohartito.scrivo.plist"
        # Scrivo was called Sotto: its Sotto.app, only when built for this folder.
        for app in /Applications/Sotto.app "$HOME/Applications/Sotto.app"; do
            if [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist" 2>/dev/null)" = org.sotto.alpha ] \
                && launcher_runs_this "$app/Contents/MacOS/Sotto"; then
                rm -r "$app"
                launchctl bootout "gui/$(id -u)/org.sotto.alpha" 2>/dev/null || true
                rm -f "$HOME/Library/LaunchAgents/org.sotto.alpha.plist"
            fi
        done
    fi
    echo "Your recordings, settings and models are still in: $DATA_DIR"
    echo "Delete that folder yourself if you want them gone."
    exit 0
fi

[ "$(uname -s)" = Darwin ] || fail "Scrivo's Mac installer runs on macOS only (see docs/windows-alpha.md for Windows)."
[ "$(uname -m)" = arm64 ] || fail "Scrivo needs an Apple Silicon Mac."
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
[ "$MACOS_MAJOR" -ge 14 ] || fail "Scrivo needs macOS 14 or later (this Mac runs $(sw_vers -productVersion))."
xcrun --find clang >/dev/null 2>&1 || fail "Install Apple's command line tools first: xcode-select --install"

# A Scrivo running from this copy would have its source and packages swapped
# while in use: a process with this copy's scrivo.py (or sotto.py, its name
# before the rename, still run by older launchers) on its command line, by its
# resolved path (what Scrivo.app runs) or relative to the process's working
# directory (a terminal run: venv-alpha/bin/python scrivo.py run). Check for
# Updates runs this from inside Scrivo, names itself in SCRIVO_UPDATE_FROM_PID
# (SOTTO_UPDATE_FROM_PID from a Scrivo started before the rename) and restarts after.
running_scrivo() {
    local listing pid command words word dir
    listing="$(ps -axww -o pid= -o command=)" || return 0
    while read -r pid command; do
        [ "$pid" != "${SCRIVO_UPDATE_FROM_PID:-${SOTTO_UPDATE_FROM_PID:-}}" ] || continue
        case " $command " in *" $ROOT/scrivo.py "*|*" $ROOT/sotto.py "*) echo "$pid"; return 0 ;; esac
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
            [ "$dir" != "$ROOT" ] || { echo "$pid"; return 0; }
        done
    done <<< "$listing"
}
RUNNING="$(running_scrivo)"
[ -z "$RUNNING" ] || fail "Scrivo is running from this copy (process $RUNNING). Quit it from its menu (or update it there with Check for Updates), then run this again."

UPSTREAM=""
REQUIREMENTS="$ROOT/requirements-alpha.txt"
if [ "$MODE" = update ]; then
    if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # Fetch first and install the new version's packages before switching the
        # source, so a failed download leaves this copy's code exactly as it was.
        git -C "$ROOT" fetch --quiet || fail "Could not download the update (git fetch failed); nothing was changed."
        UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')" || fail "This copy does not follow a GitHub branch."
        git -C "$ROOT" merge-base --is-ancestor HEAD "$UPSTREAM" \
            || fail "This copy has its own commits, so it cannot simply move forward; nothing was changed."
        NEXT_REQUIREMENTS="$(mktemp -d)"
        trap 'rm -rf "$NEXT_REQUIREMENTS"' EXIT
        for name in requirements-alpha.txt requirements.txt constraints-alpha.txt; do
            git -C "$ROOT" show "$UPSTREAM:$name" > "$NEXT_REQUIREMENTS/$name" \
                || fail "The update has no $name; nothing was changed."
        done
        REQUIREMENTS="$NEXT_REQUIREMENTS/requirements-alpha.txt"
    else
        fail "This folder is not a git clone, so it cannot update itself. Download the new source archive and run scripts/install-mac.sh from it."
    fi
fi

native_python() {  # is $1 a native arm64 Python 3.12?
    [ "$("$1" -c 'import platform; print(platform.machine())' 2>/dev/null)" = arm64 ] \
        && [ "$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = 3.12 ]
}
if [ -z "$PYTHON" ]; then
    # The first native one: an Intel Homebrew's python3.12 may come first on PATH.
    for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
        found="$(command -v "$candidate" 2>/dev/null)" || continue
        if native_python "$found"; then PYTHON="$found"; break; fi
    done
fi
[ -n "$PYTHON" ] || fail "No native arm64 Python 3.12 found. Install it (e.g. brew install python@3.12) or pass --python PATH."
[ "$("$PYTHON" -c 'import platform; print(platform.machine())')" = arm64 ] || fail "$PYTHON is not a native arm64 Python."
[ "$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.12 ] || fail "$PYTHON is not Python 3.12."

echo "== Python environment ($VENV)"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
# An exact pip, not whatever is newest today (F25); tests/test_dependency_pins.py keeps every copy in step.
"$VENV/bin/python" -m pip install --quiet pip==26.2.1
if [ -z "$UPSTREAM" ]; then
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS" \
        || fail "Installing packages failed. Run this script again to retry."
    "$VENV/bin/python" -m pip check || fail "pip check found broken requirements (see above). Run this script again to retry."
else
    # Distribution names in a pip freeze, normalized (PEP 503) and sorted.
    names() {
        sed -E -e '/^[[:space:]]*(#|-|$)/d' -e 's/[[:space:]]*[=@<>!~;[].*//' "$1" \
            | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g' | LC_ALL=C sort -u
    }
    # pip cannot undo a half-finished install: put back the set recorded before
    # it (pip install -r only adds), remove what the update added, then check it.
    restore_packages() {
        echo "== Restoring the previous packages"
        if grep -q '[^[:space:]]' "$PREVIOUS_PACKAGES"; then
            "$VENV/bin/python" -m pip install --quiet -r "$PREVIOUS_PACKAGES" || return 1
        fi
        "$VENV/bin/python" -m pip freeze > "$NEXT_REQUIREMENTS/now-packages.txt" || return 1
        local added
        added="$(LC_ALL=C comm -13 <(names "$PREVIOUS_PACKAGES") <(names "$NEXT_REQUIREMENTS/now-packages.txt"))"
        if [ -n "$added" ]; then
            echo "== Removing what the update added:" $added
            # One distribution name per word.
            # shellcheck disable=SC2086
            "$VENV/bin/python" -m pip uninstall --quiet --yes $added || return 1
        fi
        "$VENV/bin/python" -m pip check
    }
    # Any failure between the first package change and the source switch, or a
    # stop (Ctrl-C, or Check for Updates giving up), puts the packages back. A
    # second stop interrupts only the rollback's current step.
    abandon_update() {
        trap : INT TERM HUP
        if restore_packages; then
            fail "$1; the previous packages are back and the source was not switched. Run this script again to retry."
        fi
        fail "$1, and the previous packages could not be restored; the source was not switched. Run scripts/install-mac.sh --update again (it needs the network)."
    }
    PREVIOUS_PACKAGES="$NEXT_REQUIREMENTS/previous-packages.txt"
    "$VENV/bin/python" -m pip freeze > "$PREVIOUS_PACKAGES" \
        || fail "Could not list the installed packages (pip freeze failed); nothing was changed."
    trap 'abandon_update "The update was stopped"' INT TERM HUP
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS" \
        || abandon_update "Installing the update's packages failed"
    "$VENV/bin/python" -m pip check || abandon_update "The update's packages do not fit together (pip check, above)"
    echo "== Source"
    git -C "$ROOT" merge --ff-only --quiet "$UPSTREAM" \
        || abandon_update "git could not move this copy forward (local changes?)"
    trap - INT TERM HUP
fi

export SCRIVO_DATA_DIR="$DATA_DIR"
export SCRIVO_HF_HOME="$DATA_DIR/huggingface"
echo "== Speech model (first time: about 1.6 GB into $SCRIVO_HF_HOME)"
"$VENV/bin/python" "$ROOT/scrivo.py" setup

echo "== Scrivo.app"
"$VENV/bin/python" "$ROOT/scripts/install_app.py" --data-dir "$DATA_DIR"

cat <<EOF

Done. Open Scrivo like any app (Spotlight, Launchpad, or: open -a Scrivo).
The first time, allow "Scrivo" to use the Microphone and turn it on under
System Settings → Privacy & Security → Accessibility (newer macOS calls it
Device Control and Data Access). Then hold right Option, speak, and release.
Choose another key, languages and "Start Scrivo when I log in" under Settings…
in Scrivo's menu. Update later with: scripts/install-mac.sh --update
EOF

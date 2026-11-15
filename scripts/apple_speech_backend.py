"""Lazy, file-only Apple Speech adapter. Never falls back to server recognition."""
from __future__ import annotations

import threading
import time


class SpeechUnavailable(RuntimeError):
    """A fixed diagnostic code, never an NSError description or a transcript."""


class AppleSpeech:
    def __init__(self, locale: str, *, timeout: float = 120, speech=None, foundation=None):
        if speech is None or foundation is None:
            import Foundation
            import Speech

            speech, foundation = Speech, Foundation
        self.speech, self.foundation, self.timeout = speech, foundation, timeout
        if not foundation.NSThread.isMainThread():
            raise SpeechUnavailable("main-thread")
        # initWithLocale can silently fall back; require an exact supported locale.
        supported = {str(item.localeIdentifier()).replace("_", "-")
                     for item in speech.SFSpeechRecognizer.supportedLocales()}
        if locale not in supported:
            raise SpeechUnavailable("locale")
        self.recognizer = speech.SFSpeechRecognizer.alloc().initWithLocale_(
            foundation.NSLocale.alloc().initWithLocaleIdentifier_(locale))
        if self.recognizer is None or not self.recognizer.supportsOnDeviceRecognition():
            raise SpeechUnavailable("on-device")
        if not self.recognizer.isAvailable():
            raise SpeechUnavailable("unavailable")

    def _wait(self, done: threading.Event):
        deadline = time.monotonic() + self.timeout
        while not done.is_set():
            if time.monotonic() >= deadline:
                raise SpeechUnavailable("timeout")
            # Blocking the main queue with Event.wait would starve Speech callbacks.
            self.foundation.NSRunLoop.currentRunLoop().runUntilDate_(
                self.foundation.NSDate.dateWithTimeIntervalSinceNow_(0.02))

    def authorize(self):
        cls = self.speech.SFSpeechRecognizer
        status = cls.authorizationStatus()
        if status == self.speech.SFSpeechRecognizerAuthorizationStatusNotDetermined:
            # A CLI host must supply its own purpose string. Refuse before asking
            # TCC if it is missing, rather than risking macOS terminating Python.
            info = self.foundation.NSBundle.mainBundle().infoDictionary()
            if not info or not info.get("NSSpeechRecognitionUsageDescription"):
                raise SpeechUnavailable("purpose-string")
            done = threading.Event()
            statuses = []

            def authorized(value):
                statuses.append(value)
                done.set()

            cls.requestAuthorization_(authorized)
            self._wait(done)
            status = statuses[0]
        if status != self.speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
            raise SpeechUnavailable("permission")

    def transcribe(self, audio):
        # Check again for every file; availability may change during a batch.
        if not self.recognizer.supportsOnDeviceRecognition():
            raise SpeechUnavailable("on-device")
        if not self.recognizer.isAvailable():
            raise SpeechUnavailable("unavailable")
        request = self.speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(
            self.foundation.NSURL.fileURLWithPath_(str(audio)))
        request.setRequiresOnDeviceRecognition_(True)
        request.setShouldReportPartialResults_(False)
        done = threading.Event()
        state = {}

        def completed(result, error):
            if done.is_set():
                return
            try:
                if error is not None:
                    state["error"] = True
                    done.set()
                elif result is not None and result.isFinal():
                    state["text"] = str(result.bestTranscription().formattedString())
                    done.set()
            except Exception:
                # PyObjC otherwise prints callback exceptions (possibly private).
                state["error"] = True
                done.set()

        task = self.recognizer.recognitionTaskWithRequest_resultHandler_(request, completed)
        if task is None:
            raise SpeechUnavailable("recognition")
        try:
            self._wait(done)
            if state.get("error") or "text" not in state:
                raise SpeechUnavailable("recognition")
            return state["text"]
        finally:
            done.set()
            task.cancel()

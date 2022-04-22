#!/bin/bash

emacs --batch --no-init-file --load publishWeb.el --funcall toggle-debug-on-error --funcall OS-publish-all

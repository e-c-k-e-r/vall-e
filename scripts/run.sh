#!/usr/bin/env bash

until $@; do echo retrying && pkill python3; done

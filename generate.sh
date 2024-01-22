#!/usr/bin/env bash
for _ in {1..3}; do uuidgen; done | parallel --termseq INT python3 generate.py

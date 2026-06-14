#!/bin/bash

# Clean up a user-specified build directory.
cleanup() {
    DIR=$1
    rm -rf $DIR/*
}

backup() {
    for f in $(ls $1); do
        cp $1/$f /backup/$f
    done
}

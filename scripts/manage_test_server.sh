#!/bin/bash

# Function to check if the test server is running
check_server() {
    if docker-compose -f docker-compose.test.yml ps | grep -q "omeroserver.*Up"; then
        return 0
    else
        return 1
    fi
}

# Function to check if the main server is running
check_main_server() {
    if docker-compose ps | grep -q "omeroserver.*Up"; then
        return 0
    else
        return 1
    fi
}

# Function to wait for the server to be ready
wait_for_server() {
    echo "Waiting for OMERO test server to be ready..."
    for i in {1..30}; do
        if nc -z localhost 4064; then
            echo "OMERO test server is ready!"
            return 0
        fi
        sleep 2
    done
    echo "Timeout waiting for OMERO test server"
    return 1
}

case "$1" in
    start)
        if check_server; then
            echo "Test server is already running"
        else
            if check_main_server; then
                echo "Stopping main OMERO server first..."
                docker-compose down
                sleep 2
            fi
            echo "Starting test server..."
            docker-compose -f docker-compose.test.yml up -d
            wait_for_server
        fi
        ;;
    stop)
        if check_server; then
            echo "Stopping test server..."
            docker-compose -f docker-compose.test.yml down
            echo "test server stopped"
        else
            echo "Test server is not running"
        fi
        ;;
    status)
        if check_server; then
            echo "Test server is running"
        else
            echo "Test server is not running"
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac

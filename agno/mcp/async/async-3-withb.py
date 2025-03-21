import time
class SyncContextManager:
    def __enter__(self):
        print('Entering context')
        time.sleep(1)
        return self

    def __exit__(self, exc_type, exc, tb):
        print('Exiting context')
        time.sleep(1)

    def do_something(self):
        print('Doing something')
        time.sleep(1)

    def do_something2(self):
        print('Doing something again')
        time.sleep(1)

def main():
    with SyncContextManager() as manager:
        manager.do_something()
        manager.do_something2()

main()

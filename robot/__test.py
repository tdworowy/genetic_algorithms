from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def function(n):
    for i in range(1000000):
       n= n + i
    return n


if __name__ == "__main__":
    with ProcessPool() as pool:
        future = pool.map(function, range(100), timeout=10)

        iterator = future.result()

        while True:
            try:
                result = next(iterator)
                print(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
    # for i in range(100):
    #     print(function(i))
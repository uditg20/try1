import pandapower as pp
import pandapower.plotting as plot


def main() -> None:
    # Create empty network
    net = pp.create_empty_network()
    print(net)


if __name__ == "__main__":
    main()


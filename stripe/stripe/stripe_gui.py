from __future__ import division
import stripe
from enthought.enable2.wx_backend.api import Window
import remote_traits
import threading

def run_thread(stripe,quit_now):

    desired_rate = 500.0 #hz
    dt = 1.0/desired_rate

    while not quit_now.isSet():
        stripe.do_work()

def main():
    my_stripe = stripe.StripeClassWorker()
    quit_now = threading.Event()
    threading.Thread( target=run_thread, args=(my_stripe,quit_now)).start()
    my_stripe.configure_traits()
    quit_now.set()

if __name__=='__main__':
    main()


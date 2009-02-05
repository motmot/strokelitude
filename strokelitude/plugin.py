import multiprocessing
import remote_traits
import warnings

def mainloop(klass_proxy,klass_worker,
             hostname,port,obj_name,data_queue,save_data_queues,
             display_text_queue,
             quit_event):

    # the mainloop for plugins
    #print 'process %r is starting'%multiprocessing.current_process()
    instance_proxy = klass_proxy()
    server = remote_traits.ServerObj(hostname,port)
    server.serve_name(obj_name,instance_proxy)

    instance_worker = klass_worker(display_text_queue=display_text_queue,
                                   **save_data_queues)

    instance_worker.set_incoming_queue(data_queue)

    def notify_worker_func( obj, name, value ):
        setattr( instance_worker, name, value)

    instance_proxy.on_trait_change(notify_worker_func)

    if 0:
        warnings.warn('no connection wired between work and proxy')
    else:
        def notify_proxy_func( obj, name, value ):
            if hasattr( instance_proxy, name):
                setattr( instance_proxy, name, value)
        instance_worker.on_trait_change(notify_proxy_func)

    desired_rate = 500.0 #hz
    dt = 1.0/desired_rate

    while not quit_event.is_set(): # run forever
        server.handleRequests(timeout=dt) # handle network events
        instance_worker.do_work()
    #print 'process %r is shutting down'%multiprocessing.current_process()

class PluginBase(object):
    """abstract base class to create plugins for strokelitude GUI"""

    def __init__(self):
        self.quit_event = multiprocessing.Event()
        self.child = None
        self.server = None

    def get_name(self):
        raise NotImplementedError('must be overriden by derived class')
    def get_hastraits_class(self):
        raise NotImplementedError('must be overriden by derived class')

    def shutdown(self):
        self.quit_event.set()
        if self.child is not None:
            self.child.join()
            self.child = None
        if self.server is not None:
            self.server = None # close

    def startup(self):
        klass_proxy,klass_worker = self.get_hastraits_class()
        if hasattr(self,'get_worker_table_descriptions'):
            descr_dict = self.get_worker_table_descriptions()
        else:
            descr_dict = {}

        save_data_queues = {}
        for name,description in descr_dict.iteritems():
            save_data_queues[name] = multiprocessing.Queue()
        display_text_queue = multiprocessing.Queue()
        do_hostname = 'localhost'
        do_port = 8112
        if self.child is not None:
            raise ValueError('cannot start a second child')
        if self.server is not None:
            raise ValueError('cannot start a second server')

        self.quit_event.clear()
        oname = 'obj'
        data_queue = multiprocessing.Queue()
        self.child = multiprocessing.Process( target=mainloop,
                                              args=(
            klass_proxy,klass_worker,
            do_hostname,do_port,
            oname,data_queue,save_data_queues,
            display_text_queue,
            self.quit_event))
        self.child.start() # fork subprocess

        view_hostname='localhost'
        view_port = 8113
        self.server = remote_traits.ServerObj(view_hostname,view_port)
        hastraits_proxy = self.server.get_proxy_hastraits_instance(do_hostname,
                                                                   do_port,
                                                                   oname)

        return (hastraits_proxy, data_queue, save_data_queues,
                descr_dict, display_text_queue)

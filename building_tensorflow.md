# Building TensorFlow from source (TF 2.3.0, Ubuntu 20.04)

## Why build from source?
The official instructions on installing TensorFlow are here: https://www.tensorflow.org/install.
If you want to install TensorFlow just using [pip](https://www.tensorflow.org/install/pip), you are running a supported Ubuntu LTS distribution, and you're happy to install the respective tested CUDA versions (which often are outdated), by all means go ahead. A good alternative may be to run a [Docker image](https://www.tensorflow.org/install/docker).

I am usually unhappy with installing what in effect are pre-built binaries. These binaries are often not compatible with the Ubuntu version I am running, the CUDA version that I have installed, and so on. Furthermore, they may be slower than binaries optimized for the target architecture, since certain instructions are not being used (e.g. AVX2, FMA).

So installing TensorFlow from source becomes a necessity. The official instructions on building TensorFlow from source are here: https://www.tensorflow.org/install/install_sources.

What they don't mention there is that on supposedly "unsupported" configurations (i.e. up-to-date Linux systems), this can be a task from hell. So far, building TensorFlow has been a mostly terrible experience. With some TensorFlow versions and combination of system properties (OS version, CUDA version), things may work out decently. But more often than not, issues have popped up during the many times I have tried to build TensorFlow.

Trying to compile TensorFlow 2.1.0 (on Ubuntu 19.10 at the time) was particularly difficult; read about this in a [previous version](https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03/e79e25b3a847abe69c392d8ed356e58dc879234a) of this Gist. I aptly described building TensorFlow as a clusterfuck, and that may still prove true. My conservative guess is that quite a few developer years have been wasted out there because of several odd choices that have been made during TensorFlow development.

With TensorFlow 2.2.0, however, some fixes seem to have been made to improve the experience given my particular system configuration. Building went almost(!) smoothly, if one ignores the nightmare of installing CUDA and the specific Bazel requirement. At least no code and/or build file patching was required... this time around.

With CUDA 11 available "officially" for Ubuntu 20.04, another roadblock has been moved out of the way. TensorFlow 2.3.0 so far seems to be compatible with CUDA 11 & cuDNN 8.
The biggest issue that TensorFlow currently has w.r.t. building is a compilation error using GCC 10. This should be easy to fix if anyone at Google really bothered.

## Described configuration

I am describing the steps necessary to build TensorFlow in (currently) the following configuration:

 - Ubuntu 20.04
 - NVIDIA driver v450.57
 - CUDA 11.0.2 / cuDNN v8.0.2.39
 - GCC 9.3.0 (system default; Ubuntu 9.3.0-10ubuntu2)
 - TensorFlow v2.3.0

At the time of writing (**2020-08-06**), these were the latest available versions, except for the GCC version. (There seem to be [build issues](https://github.com/tensorflow/tensorflow/issues/39467) with GCC 10.1.0.)

Note that I am **not** interested in running an outdated Ubuntu version (this includes the actually quite ancient 18.04 LTS), installing a CUDA/cuDNN version that is not the latest, or using a TensorFlow version that is not the latest. Regressing to either of these is nonsensical to me. Therefore, **the below instructions may or may not be useful to you**. Please also note that the instructions **are likely outdated**, since I only update them occasionally. Many of the comments from other users below will **most certainly be outdated**. **Don't just copy these instructions, but check what the respective latest versions are and use these instead!**

## Prerequisites

### Installing the NVIDIA driver, CUDA and cuDNN
[Please refer to my instructions here](https://gist.github.com/kmhofmann/cee7c0053da8cc09d62d74a6a4c1c5e4).

### System packages
According to the [official instructions](https://www.tensorflow.org/install/source), TensorFlow requires Python and pip:

    $ sudo apt install python3-dev python3-pip python3-venv

## Installing Bazel

[Bazel](https://bazel.build/) is Google's monster of a build system and is required to build TensorFlow.

Google apparently did not want to make developers' lives easy and use a de-facto standard build system such as *CMake*. Life could be so nice. No, Google is big and dangerous enough to force their own creation upon everyone and thus make everyone else's life miserable.
I wouldn't complain if Bazel was nice and easy to use. But I don't think there were many times when I built TensorFlow and did *not* have issues with Bazel (or with a combination of the two). This may a a system that works very well inside Google, but outside of the company's infrastructure, it seems less valuable but more of a hindrance.

Anyway... things got better since a [previous version](https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03/e79e25b3a847abe69c392d8ed356e58dc879234a) of this Gist. TensorFlow 2.3.0 now requires Bazel 3.1.0, which fixes the issues encountered previously.

Note that, at the time of writing, v3.4.1 was the latest released version of Bazel. But for unfathomable reasons TensorFlow refuses to cooperate with any other version than v3.1.0. Someone at Google needs to be taught the concepts of backward or forward compatibility!

### Installing Bazel via apt
I really don't like installing things via third-party `apt` repositories, but hey, here we go. At least I hope/assume that this one is going to be decently maintained, being from the authors.
Just follow [these](https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu) instructions. The following commands should do the trick:
```
$ sudo apt install curl gnupg
$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
$ echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ sudo apt update && sudo apt install bazel-3.1.0
```

Note the explicit mention of `bazel-3.1.0` in the last line. Usually, one would just install `bazel`, but nope, TensorFlow doesn't like that as it would install the respective latest version. So it's falling back to an explicit version.

Note that using this installation mechanism for Bazel, you cannot have multiple versions of Bazel installed on your system at the same time. Here's to hoping that this will never be necessary in practice. (Just don't try to build multiple versions of Tensorflow, ever. ;-))

### Compiling Bazel from source
OK, this is much better -- we don't need to hook into the system's package management mechanism and can build completely user-locally. Official instructions [here](https://docs.bazel.build/versions/master/install-compile-source.html).

We first need to install some prerequisite dependencies:

```
sudo apt-get install build-essential openjdk-11-jdk python zip unzip
```

Since a while, Bazel needs to be built using Bazel, unless we use a bootstrapped distribution archive. These are specific per version, so let's get the right one:

```
$ wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-dist.zip
$ mkdir bazel-3.1.0
$ unzip -d ./bazel-3.1.0 bazel-3.1.0-dist.zip
```
Oh wait, we had to do... what? Yep, you read that right. If you just `unzip` the downloaded file, it will relentlessly litter the directory. Insert giant sadface here.

Also, I'd argue that it would be much easier to simply add an older version of Bazel to the repository such that user expectations are not subverted and one can follow a usual `git clone <bazel_repo_url>` followed by `make` approach. But oh well. Let's go on to call `make`.

```
$ cd bazel-3.1.0
$ env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
```

I was just joking above. Of course you don't just call `make` with Bazel. Nor do you call `cmake` and `ninja` or anything that you would expect. You call the very intuitive compilation command that is easy to figure out by yourself \</s\>.

Either way, Bazel should now be compiled. There is no equivalent to `make install`. (I don't want to repeat myself talking about user expectations, but...)
The binary should be located in `output/` relative to the unzipped Bazel source. Add that to your `PATH`.

## Building TensorFlow

### Cloning and patching

First clone the sources, and check out the desired branch. At the time of writing, `v2.2.0` was the latest version; adjust if necessary.

      $ git clone https://github.com/tensorflow/tensorflow
      $ cd tensorflow
      $ git checkout v2.3.0

### Configuration

Create a Python 3 virtual environment, if you have not done this yet. For example:

      $ python3 -m venv ~/.virtualenvs/tf_dev

Activate it with `source ~/.virtualenvs/tf_dev/bin/activate`. This can later be deactivated with `deactivate`.

Install the Python packages mentioned in the [official instructions](https://www.tensorflow.org/install/source):

    $ pip install -U pip six 'numpy<1.19.0' wheel setuptools mock 'future>=0.17.1'
    $ pip install -U keras_applications --no-deps
    $ pip install -U keras_preprocessing --no-deps

(If you choose to not use a virtual environment, you'll need to add `--user` to each of the above commands.)

Congratulations, they snuck another *maximum* version in there. This appears to be due to [this issue](https://github.com/tensorflow/tensorflow/issues/40688), but it is highly annoying nonetheless.

Now run the TensorFlow configuration script

      $ ./configure

We all like *interactive* scripts called `./configure`, don't we? (Whoever devised this atrocity has never used GNU tools before.)

Carefully go through the options. You can leave most defaults, but do specify the required CUDA compute capabilities (as below, or similar):

      CUDA support -> Y
      CUDA compute capability -> 5.2,6.1,7.0

Some of the compute capabilities of popular GPU cards might be good to know:
* Maxwell TITAN X: `5.2`
* Pascal TITAN X (2016): `6.1`
* GeForce GTX 1080 Ti: `6.1`
* Tesla V100: `7.0`

(See [here](https://developer.nvidia.com/cuda-gpus) for the full list.)

### Building

Now we can start the TensorFlow build process. 

	$ bazel build --config=opt -c opt //tensorflow/tools/pip_package:build_pip_package

Totally intuitive, right? :-D
This command will build TensorFlow using optimized settings for the current machine architecture.
    
* Add `-c dbg --strip=never` in case you do not want debug symbols to be stripped (e.g. for debugging purposes).
    Usually, you won't need to add this option.
    
* Add `--compilation_mode=dbg` to build in debug instead of release mode, i.e. without optimizations.
    You shouldn't do this unless you really want to.

This will take some time. Have a coffee, or two, or three. Cook some dinner. Watch a movie.

### Building & installing the Python package

Once the above build step has completed without error, the remainder is now easy. Build the Python package, which the `build_pip_package` script puts into a specified location.

      $ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

And install the build wheel package:

      $ pip install /tmp/tensorflow_pkg/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl

### Testing the installation

Google suggests to test the TensorFlow installation with the following command:

    $ python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

This does not make explicit use of CUDA yet, but will emit a whole bunch of initialization messages that can give an indication whether all libraries could be loaded. And it should print that requested sum.

It worked? Great! Be happy and hope you won't have to build TensorFlow again any time soon...

import re, os, time, random
import numpy as np
import gym


def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11],'envs/mujoco/assets')

    assert os.path.exists(xml_path)

    return xml_path
gym_xml_path = generate_xml_path()


def check_path(path):

    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'double' in env_name.lower():
        xml_name = "inverted_double_pendulum.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'reach' in env_name.lower():
        xml_name = "reacher.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "standup" in env_name.lower():
        xml_name = "humanoidstandup.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "striker" in env_name.lower():
        xml_name = "striker.xml"
    elif "swim" in env_name.lower():
        xml_name = "swimmer.xml"
    elif "throw" in env_name.lower():
        xml_name = "thrower.xml"
    elif "point" in env_name.lower():
        xml_name = "point.xml"
    elif "pendulum" in env_name.lower():
        xml_name = "inverted_pendulum.xml"
    elif "pusher" in env_name.lower():
        xml_name = "pusher.xml"
    elif "humanoid" in env_name.lower():
        xml_name = "humanoid.xml"
    else:
        raise RuntimeError("No available env named \'%s\'"%env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./algorithm/xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.1)


def reset_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./xml_path/init/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.1)

def save_env(env_name):
    xml_name = parse_xml_name(env_name)
    if not os.path.exists("./xml_path/init"):
        os.makedirs("./xml_path/init")

    os.system(
        'cp {1}/{0} ./xml_path/init/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.1)




def update_target_env_gravity(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree*float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "gravity=\""+replace_num+"\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.1)

def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(1)

def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree*float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\""+replace_num+"\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.1)


def update_config_files(env_name, type, degree):
    if type == "fric":
        update_target_env_friction(degree, env_name)
    elif type == "grav":
        update_target_env_gravity(degree, env_name)
    elif type == "dens":
        update_target_env_density(degree, env_name)
    else:
        raise RuntimeError("Invalid type : ", type)


class simulator():

    def __init__(self, id):

        self.env = gym.make(id)
        self.qpos_len = len(self.env.sim.data.qpos)
        self.qvel_len = len(self.env.sim.data.qvel)

        self.env.reset()

    def set_state(self, state):
        qvel = state[-self.qvel_len:]
        qpos = np.array([0] + list(state[:(self.qpos_len-1)]))

        self.env.reset()
        self.env.set_state(qpos, qvel)

        return state

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
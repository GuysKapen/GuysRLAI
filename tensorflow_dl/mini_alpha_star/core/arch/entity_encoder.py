import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from tensorflow_dl.mini_alpha_star.libs import utils
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import MiniStar_Arch_Hyper_Parameters as MAHP
from tensorflow_dl.mini_alpha_star.libs.transformer import Transformer

debug = True


def dec_to_bin(x, bits):
    mask = 2 ** tf.range(bits - 1, -1, -1, dtype=x.dtype)
    tf.math.logical_and()
    return tf.cast(tf.math.logical_and(tf.expand_dims(x, axis=-1), mask) == 0, dtype=tf.float32)


def bin_to_dec(b, bits):
    mask = 2 ** tf.range(bits - 1, -1, -1, dtype=b.dtype)
    return tf.reduce_sum(mask * b, axis=-1)


class EntityEncoder(tf.keras.Model):
    def __init__(self, dropout=.1, original_256=AHP.original_256,
                 original_1024=AHP.original_1024,
                 original_128=AHP.original_128):
        super(EntityEncoder, self).__init__()

        # below is value form max value of one-hot encoding in alphastar
        self.max_entities = AHP.max_entities
        self.max_unit_type = SCHP.max_unit_type  # default is 256
        self.max_alliance = 5

        self.max_health = 1500
        self.max_shield = 1000
        self.max_energy = 200

        self.max_cargo_space_used = 9
        self.max_cargo_space_maximum = 9

        self.max_display_type = 5  # AlphaStar: 4. RuntimeError: index 4 is out of bounds for dimension 1 with size 4
        self.max_cloakState = 5

        self.max_is_powered = 2
        self.max_is_hallucination = 2
        self.max_is_active = 2
        self.max_is_on_screen = 2
        self.max_is_in_cargo = 2

        self.max_current_minerals = 19
        self.max_current_vespene = 26

        self.max_mined_minerals = 1800
        self.max_mined_vespene = 2500

        self.max_assigned_harvesters = 25  # AlphaStar: 24. RuntimeError: index 24 is out of bounds for dimension 1 with size 24
        self.max_ideal_harvesters = 17

        self.max_weapon_cooldown = 32
        self.max_order_queue_length = 9

        self.max_order_progress = 10

        self.max_order_ids = SCHP.max_order_ids
        self.max_buffer_ids = SCHP.max_buffer_ids
        self.max_add_on_type = SCHP.max_add_on_type

        self.max_weapon_upgrades = 4
        self.max_armor_upgrades = 4
        self.max_shield_upgrades = 4

        self.max_was_selected = 2
        self.max_was_targeted = 2

        self.dropout = layers.Dropout(dropout)
        self.embedded = layers.Dense(original_256)
        self.transformer = Transformer(d_model=original_256,
                                       d_inner=original_1024,
                                       n_layers=3,
                                       n_head=2,
                                       d_k=original_128,
                                       d_v=original_128,
                                       dropout=.1)

        self.conv1 = layers.Conv1D(original_256, kernel_size=1,
                                   strides=1,
                                   padding='valid',
                                   use_bias=True)
        self.fc1 = layers.Dense(original_256)

        self.real_entities_size = 0

    def preprocess(self, entity_list):
        entity_tensor_list = []
        index = 0
        for entity in entity_list:
            field_encoding_list = []

            # unit_type: one-hot with maximum self.max_unit_type (including unknown)
            # optional unit32 unit_type=4
            # with maximum self.max_unit_type

            unit_type = entity.unit_type
            print('unit_type:', unit_type) if debug else None
            print('self.max_unit_type:', self.max_unit_type) if debug else None

            unit_type_index = utils.unit_type_to_unit_type_idx(unit_type)
            print('unit_type_index:', unit_type_index) if debug else None
            assert 0 <= unit_type_index <= self.max_unit_type

            unit_type_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([unit_type_index]), self.max_unit_type),
                shape=(1, -1)
            )
            print('unit_type_encoding:', unit_type_encoding) if debug else None
            field_encoding_list.append(unit_type_encoding)

            # A: unit_attributes: One boolean for each of the 13 unit attributes
            # B: not found
            # C: lack
            unit_attributes_encoding = tf.reshape(tf.convert_to_tensor(entity.unit_attributes, dtype=tf.float32),
                                                  shape=(1, -1))
            print(f'unit_attributes_encoding: {unit_attributes_encoding}') if debug else None
            field_encoding_list.append(unit_attributes_encoding)

            # A: alliance: One-hot with maximum 5 (including unknown alliance)
            # B: optional Alliance alliance = 2; not max is 4, not 5
            # C: use A
            alliance_encoding = tf.reshape(
                utils.one_hot_embedding(tf.convert_to_tensor([entity.alliance]),
                                        self.max_alliance),
                shape=(1, -1))
            print(f'alliance_encoding: {alliance_encoding}') if debug else None
            field_encoding_list.append(alliance_encoding)

            # A: display_type: One-hot with maximum 5
            # B: note: in s2clientprotocol raw.proto, display type only has 4 values, type of enum DisplayType,
            # C: we keep in consistent with s2clientprotocol
            display_type_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor(entity.display_type, dtype=tf.float32),
                                        self.max_display_type),
                shape=(1, -1))
            print(f'display_type_encoding: {display_type_encoding}') if debug else None
            field_encoding_list.append(display_type_encoding)

            # A: x_position: Binary encoding of entity x-coordinate, in game units
            # B: optional Point pos = 6;
            # C: use np.unpackbits
            x = entity.x
            x_encoding = tf.reshape(
                tf.convert_to_tensor(np.unpackbits(np.array([x], np.uint8)), dtype=tf.float32), shape=(1, -1))
            print(f'x_encoding: {x_encoding}') if debug else None
            field_encoding_list.append(x_encoding)

            # A: y_position: Binary encoding of entity y-coordinate, in game units
            # B: optional Point pos = 6;
            # C: use np.unpackbits
            # change sequence due to easy processing of entity_x_y_index
            y = entity.y
            y_encoding = tf.reshape(
                tf.convert_to_tensor(np.unpackbits(np.array([y], np.uint8)), dtype=tf.float32), shape=(1, -1))
            print(f'y_encoding: {y_encoding}') if debug else None
            field_encoding_list.append(y_encoding)

            # A: current_health: One-hot of sqrt(min(current_health, 1500)) with maximum sqrt(1500), rounding down
            # B: optional float health = 14;
            # C: None
            print('entity.health:', entity.health) if debug else None
            current_health = int(min(entity.health, self.max_health) ** 0.5)
            print('current_health:', current_health) if debug else None
            current_health_encoding = tf.reshape(utils.to_one_hot(tf.convert_to_tensor([current_health]),
                                                                  int(self.max_health ** 0.5) + 1), shape=(1, -1))
            print('current_health_encoding.shape:', current_health_encoding.shape) if debug else None
            field_encoding_list.append(current_health_encoding)

            # A: current_shields: One-hot of sqrt(min(current_shields, 1000)) with maximum sqrt(1000), rounding down
            # B: optional float shield = 16;
            # C: None
            print('entity.shield:', entity.shield) if debug else None
            current_shield = int(min(entity.shield, self.max_shield) ** 0.5)
            print('current_shield:', current_shield) if debug else None
            current_shield_encoding = tf.reshape(
                utils.to_one_hot(
                    tf.convert_to_tensor([current_shield]), int(self.max_shield ** 0.5) + 1),
                shape=(1, -1))
            print('current_shield_encoding.shape:', current_shield_encoding.shape) if debug else None
            field_encoding_list.append(current_shield_encoding)

            # A: current_energy: One-hot of sqrt(min(current_energy, 200)) with maximum sqrt(200), rounding down
            # B: optional float energy = 17;
            # C: None
            print('entity.energy:', entity.energy) if debug else None
            current_energy = int(min(entity.energy, self.max_energy) ** 0.5)
            print('current_energy:', current_energy) if debug else None
            current_energy_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([current_energy]), int(self.max_energy ** 0.5) + 1),
                shape=(1, -1))
            print('current_energy_encoding.shape:', current_energy_encoding.shape) if debug else None
            field_encoding_list.append(current_energy_encoding)

            # A: cargo_space_used: One-hot with maximum 9
            # B: optional int32 cargo_space_taken = 25;
            # C: None
            cargo_space_used = entity.cargo_space_taken
            assert 0 <= cargo_space_used <= 8
            cargo_space_used_encoding = tf.reshape(
                utils.to_one_hot(
                    tf.convert_to_tensor([cargo_space_used]), self.max_cargo_space_used),
                shape=(1, -1))
            print('cargo_space_used_encoding:', cargo_space_used_encoding) if debug else None
            field_encoding_list.append(cargo_space_used_encoding)

            # A: cargo_space_maximum: One-hot with maximum 9
            # B: optional int32 cargo_space_max = 26;
            # C: None
            cargo_space_maximum = entity.cargo_space_max
            assert 0 <= cargo_space_maximum <= 8
            cargo_space_maximum_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([cargo_space_maximum]), self.max_cargo_space_maximum),
                shape=(1, -1))
            print('cargo_space_maximum_encoding:', cargo_space_maximum_encoding) if debug else None
            field_encoding_list.append(cargo_space_maximum_encoding)

            # A: build_progress: Float of build progress, in [0, 1]
            # B: optional float build_progress = 9;        // Range: [0.0, 1.0]
            # C: None
            build_progress_encoding = tf.reshape(tf.convert_to_tensor([entity.build_progress], dtype=tf.float32),
                                                 shape=(1, -1))
            print('build_progress_encoding:', build_progress_encoding) if debug else None
            field_encoding_list.append(build_progress_encoding)

            # A: current_health_ratio: Float of health ratio, in [0, 1]
            # B: optional float health_max = 15;
            # C: None
            current_health_ratio = entity.health / entity.health_max if entity.health_max else 0.
            current_health_ratio_encoding = tf.reshape(tf.convert_to_tensor([current_health_ratio], dtype=tf.float32),
                                                       shape=(1, -1))
            print('current_health_ratio_encoding:', current_health_ratio_encoding) if debug else None
            field_encoding_list.append(current_health_ratio_encoding)

            # A: current_shield_ratio: Float of shield ratio, in [0, 1]
            # B: optional float shield_max = 36;
            # C: None
            current_shield_ratio = entity.shield / entity.shield_max if entity.shield_max else 0.
            current_shield_ratio_encoding = tf.reshape(tf.convert_to_tensor([current_shield_ratio], dtype=tf.float32),
                                                       shape=(1, -1))
            print('current_shield_ratio_encoding:', current_shield_ratio_encoding) if debug else None
            field_encoding_list.append(current_shield_ratio_encoding)

            # A: current_energy_ratio: Float of energy ratio, in [0, 1]
            # B: optional float energy_max = 37;
            # C: None
            current_energy_ratio = entity.energy / entity.energy_max if entity.energy_max else 0.
            current_energy_ratio_encoding = tf.reshape(tf.convert_to_tensor([current_energy_ratio], dtype=tf.float32),
                                                       shape=(1, -1))
            print('current_energy_ratio_encoding:', current_energy_ratio_encoding) if debug else None
            field_encoding_list.append(current_energy_ratio_encoding)

            # A: is_cloaked: One-hot with maximum 5
            # B: note: in s2clientprotocol raw.proto, this is called clock, type of enum CloakState,
            # C: we keep in consistent with s2clientprotocol
            # clock_value = entity.cloak.value
            cloak_state_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([entity.is_cloaked]), self.max_cloakState),
                shape=(1, -1))
            print('cloakState_encoding:', cloak_state_encoding) if debug else None
            field_encoding_list.append(cloak_state_encoding)

            # A: is_powered: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int
            is_powered_value = int(entity.is_powered)
            is_powered_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([is_powered_value]), self.max_is_powered),
                shape=(1, -1))
            print('is_powered_encoding:', is_powered_encoding) if debug else None
            field_encoding_list.append(is_powered_encoding)

            # A: is_hallucination: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int
            is_hallucination_value = int(entity.is_hallucination)
            is_hallucination_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([is_hallucination_value]), self.max_is_hallucination),
                shape=(1, -1))
            print('is_hallucination_encoding:', is_hallucination_encoding) if debug else None
            field_encoding_list.append(is_hallucination_encoding)

            # A: is_active: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int
            is_active_value = int(entity.is_active)
            is_active_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([is_active_value]), self.max_is_active),
                shape=(1, -1))
            print('is_active_encoding:', is_active_encoding) if debug else None
            field_encoding_list.append(is_active_encoding)

            # A: is_on_screen: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, this is type of bool
            # C: we convert it from bool to int
            is_on_screen_value = int(entity.is_on_screen)
            is_on_screen_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([is_on_screen_value]), self.max_is_on_screen),
                shape=(1, -1))
            print('is_on_screen_encoding:', is_on_screen_encoding) if debug else None
            field_encoding_list.append(is_on_screen_encoding)

            # A: is_in_cargo: One-hot with maximum 2
            # B: note: in s2clientprotocol raw.proto, there is no is_in_cargo
            # C: wait to be resolved by other ways
            is_in_cargo_value = int(entity.is_in_cargo)
            is_in_cargo_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([is_in_cargo_value]), self.max_is_in_cargo),
                shape=(1, -1))
            print('is_in_cargo_encoding:', is_in_cargo_encoding) if debug else None
            field_encoding_list.append(is_in_cargo_encoding)

            # A: current_minerals: One-hot of (current_minerals / 100) with maximum 19, rounding down
            # B: optional int32 mineral_contents = 18; (maybe)
            # C: I am not sure mineral_contents corresponds to current_minerals
            print('entity.current_minerals:', entity.current_minerals) if debug else None
            current_minerals = int(entity.current_minerals / 100)
            print('current_minerals:', current_minerals) if debug else None
            current_minerals_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([current_minerals]), self.max_current_minerals),
                shape=(1, -1))
            print('current_minerals_encoding.shape:', current_minerals_encoding.shape) if debug else None
            field_encoding_list.append(current_minerals_encoding)

            # A: current_vespene: One-hot of (current_vespene / 100) with maximum 26, rounding down
            # B: optional int32 vespene_contents = 19; (maybe)
            # C: I am not sure vespene_contents corresponds to current_vespene
            print('entity.current_vespene:', entity.current_vespene) if debug else None
            current_vespene = int(entity.current_vespene / 100)
            print('current_vespene:', current_vespene) if debug else None
            current_vespene_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([current_vespene]), self.max_current_vespene),
                shape=(1, -1))
            print('current_vespene_encoding.shape:', current_vespene_encoding.shape) if debug else None
            field_encoding_list.append(current_vespene_encoding)

            # A: mined_minerals: One-hot of sqrt(min(mined_minerals, 1800)) with maximum sqrt(1800), rounding down
            # B: not found
            # C: wait to be resolved by other ways
            print('entity.mined_minerals:', entity.mined_minerals) if debug else None
            mined_minerals = int(min(entity.mined_minerals, self.max_mined_minerals) ** 0.5)
            print('mined_minerals:', mined_minerals) if debug else None
            mined_minerals_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([mined_minerals]), int(self.max_mined_minerals ** 0.5) + 1),
                shape=(1, -1))
            print('mined_minerals_encoding.shape:', mined_minerals_encoding.shape) if debug else None
            field_encoding_list.append(mined_minerals_encoding)

            # A: mined_vespene: One-hot of sqrt(min(mined_vespene, 2500)) with maximum sqrt(2500), rounding down
            # B: not found
            # C: wait to be resolved by other ways
            print('entity.mined_vespene:', entity.mined_vespene) if debug else None
            mined_vespene = int(min(entity.mined_vespene, self.max_mined_vespene) ** 0.5)
            print('mined_vespene:', mined_vespene) if debug else None
            mined_vespene_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([mined_vespene]), int(self.max_mined_vespene ** 0.5) + 1),
                shape=(1, -1))
            print('mined_vespene_encoding.shape:', mined_vespene_encoding.shape) if debug else None
            field_encoding_list.append(mined_vespene_encoding)

            # A: assigned_harvesters: One-hot with maximum 24
            # B: optional int32 assigned_harvesters = 28;
            # C: None
            assigned_harvesters_encoding = tf.reshape(utils.to_one_hot(
                tf.convert_to_tensor([entity.assigned_harvesters]), self.max_assigned_harvesters),
                shape=(1, -1))
            print('assigned_harvesters_encoding:', assigned_harvesters_encoding) if debug else None
            field_encoding_list.append(assigned_harvesters_encoding)

            # A: ideal_harvesters: One-hot with maximum 17
            # B: optional int32 ideal_harvesters = 29;
            # C: None
            ideal_harvesters_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([entity.ideal_harvesters]), self.max_ideal_harvesters),
                shape=(1, -1))
            print('ideal_harvesters_encoding:', ideal_harvesters_encoding) if debug else None
            field_encoding_list.append(ideal_harvesters_encoding)

            # A: weapon_cooldown: One-hot with maximum 32 (counted in game steps)
            # B: optional float weapon_cooldown = 30;
            # C: None
            weapon_cooldown = int(entity.weapon_cooldown)
            weapon_cooldown_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([weapon_cooldown]), self.max_weapon_cooldown),
                shape=(1, -1))
            print('weapon_cooldown_encoding:', weapon_cooldown_encoding) if debug else None
            field_encoding_list.append(weapon_cooldown_encoding)

            # A: order_queue_length: One-hot with maximum 9
            # B: repeated UnitOrder orders = 22; Not populated for enemies;
            # C: equal to FeatureUnit.order_length
            order_queue_length = entity.order_length
            order_queue_length_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([order_queue_length]), self.max_order_queue_length),
                shape=(1, -1))
            print('order_queue_length_encoding:', order_queue_length_encoding) if debug else None
            field_encoding_list.append(order_queue_length_encoding)

            # A: order_1: One-hot across all order IDs
            # B: below is the definition of order
            '''
                message UnitOrder {
                      optional uint32 ability_id = 1;
                      one of target {
                        Point target_world_space_pos = 2;
                        uint64 target_unit_tag = 3;
                      }
                      optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
                    }
            '''
            # C: actually this is across all ability_ids in orders, lack: a vector for all ability_ids
            order_1 = entity.order_id_1
            order_1_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([order_1]), self.max_order_ids),
                shape=(1, -1))
            print('order_1_encoding:', order_1_encoding) if debug else None
            field_encoding_list.append(order_1_encoding)

            # A: order_2: One-hot across all building training order IDs. Note that this tracks queued building orders, and unit orders will be ignored
            # B: None
            # C: in mAS, we ingore it
            if AHP != MAHP:
                order_2 = entity.order_id_2
                order_2_encoding = tf.reshape(
                    utils.to_one_hot(tf.convert_to_tensor([order_2]), self.max_order_ids),
                    shape=(1, -1))
                print('order_2_encoding:', order_2_encoding) if debug else None
                field_encoding_list.append(order_2_encoding)

            # A: order_3: One-hot across all building training order IDs
            # B: None
            # C: in mAS, we ignore it
            if AHP != MAHP:
                order_3 = entity.order_id_3
                order_3_encoding = tf.reshape(
                    utils.to_one_hot(tf.convert_to_tensor([order_3]), self.max_order_ids),
                    shape=(1, -1))
                print('order_3_encoding:', order_3_encoding) if debug else None
                field_encoding_list.append(order_3_encoding)

            # A: order_4: One-hot across all building training order IDs
            # B: None
            # C: in mAS, we ignore it
            if AHP != MAHP:
                order_4 = entity.order_id_4
                order_4_encoding = tf.reshape(
                    utils.to_one_hot(tf.convert_to_tensor([order_4]), self.max_order_ids),
                    shape=(1, -1))
                print('order_4_encoding:', order_4_encoding) if debug else None
                field_encoding_list.append(order_4_encoding)

            # A: buffs: Boolean for each buff of whether or not it is active. Only the first two buffs are tracked
            # B: None
            # C: in mAS, we ingore buff_id_2
            buff_id_1 = entity.buff_id_1
            buff_id_1_encoding = tf.reshape(utils.to_one_hot(tf.convert_to_tensor([buff_id_1]), self.max_buffer_ids),
                                            shape=(1, -1))
            print('buff_id_1_encoding:', buff_id_1_encoding) if debug else None
            field_encoding_list.append(buff_id_1_encoding)

            if AHP != MAHP:
                buff_id_2 = entity.buff_id_2
                buff_id_2_encoding = tf.reshape(
                    utils.to_one_hot(tf.convert_to_tensor([buff_id_2]), self.max_buffer_ids), shape=(1, -1))
                print('buff_id_2_encoding:', buff_id_2_encoding) if debug else None
                field_encoding_list.append(buff_id_2_encoding)

            # TODO (Important!): Move the order_length, 4 orders and 2 buff ids and order progresses to the field_encoding_list

            # A: addon_type: One-hot of every possible add-on type
            # B: optional uint64 add_on_tag = 23;
            # C: lack: from tag to find a unit, then get the unit type of the unit
            # in mAS, we ingore it
            if AHP != MAHP:
                addon_unit_type = entity.addon_unit_type
                addon_unit_type_encoding = tf.reshape(utils.to_one_hot(tf.convert_to_tensor([addon_unit_type]),
                                                                       self.max_add_on_type), shape=(1, -1))
                print('addon_unit_type_encoding:', addon_unit_type_encoding) if debug else None
                field_encoding_list.append(addon_unit_type_encoding)

            # A: order_progress_1: Float of order progress, in [0, 1], and one-hot of (`order_progress_1` / 10) with maximum 10
            # B: optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
            # C: None
            '''
            orders = entity.orders
            if len(orders) >= 1:
                order_1 = orders[0]
                if hasattr(order_1, 'progress'):
            '''

            order_progress_1_encoding = tf.zeros((1, 1), dtype=tf.float32)
            order_progress_1_encoding_2 = tf.zeros((1, self.max_order_progress), dtype=tf.float32)
            order_progress_1 = entity.order_progress_1
            if order_progress_1 is not None:
                order_progress_1_encoding = tf.reshape(
                    tf.convert_to_tensor([order_progress_1 / 100.], dtype=tf.float32), shape=(1, -1))
                order_progress_1_encoding_2 = tf.reshape(utils.to_one_hot(tf.convert_to_tensor([order_progress_1 / 10]),
                                                                          self.max_order_progress), shape=(1, -1))
            print('order_progress_1_encoding:', order_progress_1_encoding) if debug else None
            field_encoding_list.append(order_progress_1_encoding)
            print('order_progress_1_encoding_2:', order_progress_1_encoding_2) if debug else None
            field_encoding_list.append(order_progress_1_encoding_2)

            # A: order_progress_2: Float of order progress, in [0, 1], and one-hot of (`order_progress_2` / 10) with maximum 10
            # B: optional float progress = 4;              // Progress of train abilities. Range: [0.0, 1.0]
            # C: None
            '''
            if len(orders) >= 2:
                order_2 = orders[1]
                if hasattr(order_2, 'progress'):
            '''

            order_progress_2_encoding = tf.zeros((1, 1), dtype=tf.float32)
            order_progress_2_encoding_2 = tf.zeros((1, self.max_order_progress), dtype=tf.float32)
            order_progress_2 = entity.order_progress_2
            if order_progress_2 is not None:
                order_progress_2_encoding = tf.reshape(
                    tf.convert_to_tensor([order_progress_2 / 100.], dtype=tf.float32), shape=(1, -1))
                order_progress_2_encoding_2 = tf.reshape(
                    utils.to_one_hot(tf.convert_to_tensor([order_progress_2 / 10]), self.max_order_progress),
                    shape=(1, -1))
            print('order_progress_2_encoding:', order_progress_2_encoding) if debug else None
            field_encoding_list.append(order_progress_2_encoding)
            print('order_progress_2_encoding_2:', order_progress_2_encoding_2) if debug else None
            field_encoding_list.append(order_progress_2_encoding_2)

            # A: weapon_upgrades: One-hot with maximum 4
            # B: optional int32 attack_upgrade_level = 40;
            # C: None
            weapon_upgrades = entity.attack_upgrade_level
            assert 0 <= weapon_upgrades <= 3
            weapon_upgrades_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([weapon_upgrades]), self.max_weapon_upgrades), shape=(1, -1))
            print('weapon_upgrades_encoding:', weapon_upgrades_encoding) if debug else None
            field_encoding_list.append(weapon_upgrades_encoding)

            # A: armor_upgrades: One-hot with maximum 4
            # B: optional int32 armor_upgrade_level = 41;
            # C: None
            armor_upgrades = entity.armor_upgrade_level
            assert 0 <= armor_upgrades <= 3
            armor_upgrades_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([armor_upgrades]), self.max_armor_upgrades),
                shape=(1, -1))
            print('armor_upgrades_encoding:', armor_upgrades_encoding) if debug else None
            field_encoding_list.append(armor_upgrades_encoding)

            # A: shield_upgrades: One-hot with maximum 4
            # B: optional int32 armor_upgrade_level = 41;
            # C: None
            shield_upgrades = entity.shield_upgrade_level
            assert 0 <= shield_upgrades <= 3
            shield_upgrades_encoding = tf.reshape(utils.to_one_hot(
                tf.convert_to_tensor([shield_upgrades]), self.max_shield_upgrades), shape=(1, -1))
            print('shield_upgrades_encoding:', shield_upgrades_encoding) if debug else None
            field_encoding_list.append(shield_upgrades_encoding)

            # A: was_selected: One-hot with maximum 2 of whether this unit was selected last actionn
            # B: optional bool is_selected = 11;
            # C: None
            was_selected = int(entity.is_selected)
            was_selected_encoding = tf.reshape(
                utils.to_one_hot(tf.convert_to_tensor([was_selected]), self.max_was_selected),
                shape=(1, -1))
            print('was_selected_encoding:', was_selected_encoding) if debug else None
            field_encoding_list.append(was_selected_encoding)

            # A: was_targeted: One-hot with maximum 2 of whether this unit was targeted last action
            # B: not found,   optional uint64 engaged_target_tag = 34; (maybe)
            # C: None
            was_targeted = int(entity.is_targeted)
            was_targeted_encoding = tf.reshape(utils.to_one_hot(
                tf.convert_to_tensor([was_targeted]), self.max_was_targeted), shape=(1, -1))
            print('was_targeted_encoding:', was_targeted_encoding) if debug else None
            field_encoding_list.append(was_targeted_encoding)

            entity_tensor = tf.concat(field_encoding_list, axis=1)
            print(f'entity_tensor.shape: {entity_tensor.shape}') if debug else None

            # There are up to 512 of these preprocessed entities, and any entities after 512 are ignored
            if index < self.max_entities:
                entity_tensor_list.append(entity_tensor)
            else:
                break
            index = index + 1

        all_entities_tensor = tf.concat(entity_tensor_list, axis=0)
        # real entities
        self.real_entities_size = all_entities_tensor.shape[0]
        print(f'self.real_entities_size: {self.real_entities_size}') if debug else None

        # Use bias of -1e9 for any of the 512 entries that doesn't refer to an entity
        if all_entities_tensor.shape[0] < self.max_entities:
            bias_len = self.max_entities - all_entities_tensor.shape[0]
            bias = tf.fill((bias_len, AHP.embedding_size), -1e9)
            print(f'bias: {bias}') if debug else None
            print(f'bias.shape: {bias.shape}') if debug else None
            all_entities_tensor = tf.concat([all_entities_tensor, bias], axis=0)

        return all_entities_tensor

    def call(self, inputs, training=None, mask=None):
        # assert the input shape is : batch_seq_size x entities_size x embeding_size
        # note: because the feature size of entity is not equal to 256, so it can not fed into transformer directly.
        # thus, we add a embedding layer to transfer it to right size.
        x = self.embedded(inputs)

        # x is batch_entities_tensor (n_dim = 3) b x entities_size x embedding_size
        # after: x is batch_seq_entities_tensor (n_dim = 4) b x seq_size x entities_size x embedding_size
        print(f'x.shape: {x.shape}') if debug else None
        out = self.transformer(x)
        print(f'out.shape: {out.shape}') if debug else None
        entity_embeddings = tf.nn.relu(self.conv1(tf.nn.relu(out)))
        print(f'entity_embeddings.shape: {entity_embeddings.shape}') if debug else None

        # masked by the missing entries
        print(f'out.shape: {out.shape}') if debug else None
        out = out[:, :self.real_entities_size, :]
        print(f'out.shape: {out.shape}') if debug else None

        # note axis=1 means the mean is across all entities in one timestep
        # the mean of the transformer output across the units is feed through a linear layer of size 256 and relu act
        # to yield embedded entities
        embedded_entities = tf.nn.relu(self.fc1(tf.reduce_mean(out, axis=1, keepdims=False)))
        print(f'embedded_entity.shape: {embedded_entities.shape}') if debug else None

        return entity_embeddings, embedded_entities


class Entity(object):

    def __init__(self, unit_type=1,
                 unit_attributes=(0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0), alliance=0,
                 health=10, shield=20, energy=50,
                 cargo_space_taken=0, cargo_space_max=0, build_progress=0,
                 current_health_ratio=0.4, current_shield_ratio=0.5, current_energy_ratio=0.7,
                 health_max=100, shield_max=50, energy_max=40,
                 display_type=1, x=123, y=218, is_cloaked=3, is_powered=True, is_hallucination=False, is_active=True,
                 is_on_screen=True, is_in_cargo=False, current_minerals=1000, current_vespene=1500, mined_minerals=500,
                 mined_vespene=300, assigned_harvesters=8, ideal_harvesters=14, weapon_cooldown=5.0,
                 orders=(0, 1, 3, 0),
                 attack_upgrade_level=2, armor_upgrade_level=1, shield_upgrade_level=0, is_selected=True,
                 is_targeted=False,
                 order_length=4, order_id_0=1, order_id_1=0, order_id_2=3, order_id_3=2, order_progress_0=50,
                 order_progress_1=95, buff_id_0=12, buff_id_1=8, addon_unit_type=4, tag=0):
        super().__init__()
        self.unit_type = unit_type
        self.unit_attributes = unit_attributes
        self.alliance = alliance
        self.health = health
        self.shield = shield
        self.energy = energy
        self.cargo_space_taken = cargo_space_taken
        self.cargo_space_max = cargo_space_max
        self.build_progress = build_progress
        self.current_health_ratio = current_health_ratio
        self.current_shield_ratio = current_shield_ratio
        self.current_energy_ratio = current_energy_ratio
        self.health_max = health_max
        self.shield_max = shield_max
        self.energy_max = energy_max
        self.display_type = display_type
        self.x = x
        self.y = y
        self.is_cloaked = is_cloaked
        self.is_powered = is_powered
        self.is_hallucination = is_hallucination
        self.is_active = is_active
        self.is_on_screen = is_on_screen
        self.is_in_cargo = is_in_cargo
        self.current_minerals = current_minerals
        self.current_vespene = current_vespene
        self.mined_minerals = mined_minerals
        self.mined_vespene = mined_vespene
        self.assigned_harvesters = assigned_harvesters
        self.ideal_harvesters = ideal_harvesters
        self.weapon_cooldown = weapon_cooldown
        self.attack_upgrade_level = attack_upgrade_level
        self.armor_upgrade_level = armor_upgrade_level
        self.shield_upgrade_level = shield_upgrade_level
        self.is_selected = is_selected
        self.is_targeted = is_targeted
        self.order_length = order_length
        self.order_id_1 = order_id_0
        self.order_id_2 = order_id_1
        self.order_id_3 = order_id_2
        self.order_id_4 = order_id_3
        self.order_progress_1 = order_progress_0
        self.order_progress_2 = order_progress_1
        self.buff_id_1 = buff_id_0
        self.buff_id_2 = buff_id_1
        self.addon_unit_type = addon_unit_type
        self.tag = tag

    def __str__(self):
        return 'unit_type: ' + str(self.unit_type) + ', alliance: ' + str(self.alliance) + ', health: ' + str(
            self.health)


def test():
    print(tf.convert_to_tensor(np.unpackbits(np.array([25], np.uint8))))
    batch_size = 2

    e_list = []
    e1 = Entity(115, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 0, 100, 60, 50, 4, 8, 95, 0.2, 0.0, 0.0, 140, 60, 100,
                1, 123, 218, 3, True, False, True, True, False, 0, 0, 0, 0, 0, 0, 3.0, [2, 3], 2, 1, 0, True, False)
    e2 = Entity(1908, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 1500, 0, 200, 0, 4, 15, 0.5, 0.8, 0.5, 1500, 0, 250,
                2, 69, 7, 3, True, False, False, True, False, 0, 0, 0, 0, 10, 16, 0.0, [1], 1, 1, 0, False, False)
    e_list.append(e1)
    e_list.append(e2)

    encoder = EntityEncoder()
    entities_tensor = encoder.preprocess(e_list)
    print(f'entities_tensor: {entities_tensor}') if debug else None
    print(f'entities_tensor.shape: {entities_tensor.shape}') if debug else None
    # entities_tensor n_dim = 2, entities_size x embedding_size

    entities_tensor = tf.expand_dims(entities_tensor, axis=0)
    if batch_size == 2:
        entities_tensor_copy = tf.stop_gradient(entities_tensor)
        batch_entities_tensor = tf.concat([entities_tensor, entities_tensor_copy], axis=0)

    print(f'batch_entities_tensor.shape: {batch_entities_tensor.shape}') if debug else None
    entity_embeddings, embedded_entity = encoder(batch_entities_tensor)

    print(f'entity_embeddings.shape: {entity_embeddings.shape}') if debug else None
    print(f'embedded_entity.shape: {embedded_entity.shape}') if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()

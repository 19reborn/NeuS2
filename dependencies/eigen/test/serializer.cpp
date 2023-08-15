// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <vector>
#include <Eigen/Core>

struct MyPodType {
  double x;
  int y;
  float z;
};

// Plain-old-data serialization.
void test_pod_type() {
  MyPodType initial = {1.3, 17, 1.9f};
  MyPodType clone = {-1, -1, -1};
  
  Eigen::Serializer<MyPodType> serializer;
  
  // Determine required size.
  size_t buffer_size = serializer.size(initial);
  VERIFY_IS_EQUAL(buffer_size, sizeof(MyPodType));
  
  // Serialize.
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* dest = serializer.serialize(buffer.data(), initial);
  VERIFY_IS_EQUAL(dest - buffer.data(), buffer_size);
  
  // Deserialize.
  uint8_t* src = serializer.deserialize(buffer.data(), clone);
  VERIFY_IS_EQUAL(src - buffer.data(), buffer_size);
  VERIFY_IS_EQUAL(clone.x, initial.x);
  VERIFY_IS_EQUAL(clone.y, initial.y);
  VERIFY_IS_EQUAL(clone.z, initial.z);
}

// Matrix, Vector, Array
template<typename T>
void test_eigen_type(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();
  
  const T initial = T::Random(rows, cols);
  
  // Serialize.
  Eigen::Serializer<T> serializer;
  size_t buffer_size = serializer.size(initial);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* dest = serializer.serialize(buffer.data(), initial);
  VERIFY_IS_EQUAL(dest - buffer.data(), buffer_size);
  
  // Deserialize.
  T clone;
  uint8_t* src = serializer.deserialize(buffer.data(), clone);
  VERIFY_IS_EQUAL(src - buffer.data(), buffer_size);
  VERIFY_IS_CWISE_EQUAL(clone, initial);
}

// Test a collection of dense types.
template<typename T1, typename T2, typename T3>
void test_dense_types(const T1& type1, const T2& type2, const T3& type3) {
  
  // Make random inputs.
  const T1 x1 = T1::Random(type1.rows(), type1.cols());
  const T2 x2 = T2::Random(type2.rows(), type2.cols());
  const T3 x3 = T3::Random(type3.rows(), type3.cols());
  
  // Allocate buffer and serialize.
  size_t buffer_size = Eigen::serialize_size(x1, x2, x3);
  std::vector<uint8_t> buffer(buffer_size);
  Eigen::serialize(buffer.data(), x1, x2, x3);
  
  // Clone everything.
  T1 y1;
  T2 y2;
  T3 y3;
  Eigen::deserialize(buffer.data(), y1, y2, y3);
  
  // Verify they equal.
  VERIFY_IS_CWISE_EQUAL(y1, x1);
  VERIFY_IS_CWISE_EQUAL(y2, x2);
  VERIFY_IS_CWISE_EQUAL(y3, x3);
}

EIGEN_DECLARE_TEST(serializer)
{
  CALL_SUBTEST( test_pod_type() );

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( test_eigen_type(Eigen::Array33f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::ArrayXd(10)) );
    CALL_SUBTEST( test_eigen_type(Eigen::Vector3f()) );
    CALL_SUBTEST( test_eigen_type(Eigen::Matrix4d()) );
    CALL_SUBTEST( test_eigen_type(Eigen::MatrixXd(15, 17)) );
    
    CALL_SUBTEST( test_dense_types( Eigen::Array33f(),
                                    Eigen::ArrayXd(10),
                                    Eigen::MatrixXd(15, 17)) );
  }
}

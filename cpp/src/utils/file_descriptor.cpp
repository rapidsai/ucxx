/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <cstdio>
#include <exception>
#include <ios>
#include <string>

#include <ucxx/utils/file_descriptor.h>

namespace ucxx {

namespace utils {

FILE* createTextFileDescriptor()
{
  FILE* textFileDescriptor = std::tmpfile();
  if (textFileDescriptor == nullptr) throw std::ios_base::failure("tmpfile() failed");

  return textFileDescriptor;
}

std::string decodeTextFileDescriptor(FILE* textFileDescriptor)
{
  size_t size;

  rewind(textFileDescriptor);
  fseek(textFileDescriptor, 0, SEEK_END);
  size = ftell(textFileDescriptor);
  rewind(textFileDescriptor);

  std::string textString(size, '\0');

  if (fread(&textString[0], sizeof(char), size, textFileDescriptor) != size)
    throw std::ios_base::failure("fread() failed");

  fclose(textFileDescriptor);

  return textString;
}

}  // namespace utils

}  // namespace ucxx
